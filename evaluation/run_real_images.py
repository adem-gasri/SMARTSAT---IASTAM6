"""
evaluation/run_real_images.py

Runs the SmartSat pipeline (tiling + YOLO + NMS + confidence-aware payload)
on a folder of CLEAN validation images and compares the transmission
strategies on REAL detections, against the ground-truth annotations.

Payload sizes are MEASURED: every transmitted crop is really cropped from
the image and JPEG-encoded in memory (the old fixed sizes are kept as
"modeled" values for comparison).

Usage:
    python evaluation/run_real_images.py --images detection/test_images/images --labels detection/test_images/labels --weights detection/models/best_v2.pt --tile-size 768
"""

import argparse
import csv
import io
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from spatial_dedup.tiling import generate_tiles, tile_box_to_global      # noqa: E402
from spatial_dedup.nms_merge import nms_merge, iou                       # noqa: E402
from evaluation.baselines import (                                       # noqa: E402
    METADATA_SIZE_BYTES, confidence_tier, payload_size_for_detection,
)

IMG_EXTS = (".jpg", ".jpeg", ".png")
TIERS = ["high_confidence", "uncertain", "highly_uncertain"]


def load_gt_boxes(label_path, img_w, img_h):
    """Read a YOLO label file and return boxes as [x1, y1, x2, y2] in pixels."""
    boxes = []
    if not label_path or not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            _, xc, yc, w, h = map(float, parts[:5])
            boxes.append([(xc - w / 2) * img_w, (yc - h / 2) * img_h,
                          (xc + w / 2) * img_w, (yc + h / 2) * img_h])
    return boxes


def match_flags(detections, gt_boxes, iou_thr=0.5):
    """Greedy one-to-one matching, highest confidence first.
    Returns a list of booleans (True = true positive) aligned with `detections`."""
    order = sorted(range(len(detections)), key=lambda i: detections[i]["confidence"], reverse=True)
    used, flags = set(), [False] * len(detections)
    for i in order:
        best_j, best_iou = None, iou_thr
        for j, g in enumerate(gt_boxes):
            if j in used:
                continue
            v = iou(detections[i]["bbox_xyxy"], g)
            if v >= best_iou:
                best_iou, best_j = v, j
        if best_j is not None:
            used.add(best_j)
            flags[i] = True
    return flags


def covered_gt(detections, gt_boxes, iou_thr=0.5):
    """Number of true ships overlapped (IoU >= thr) by at least one detection."""
    return sum(1 for g in gt_boxes if any(iou(d["bbox_xyxy"], g) >= iou_thr for d in detections))


def crop_with_margin(img, bbox, margin=0.2, min_side=4):
    """Crop the detection box plus a 20% margin, clamped to the image."""
    x1, y1, x2, y2 = bbox
    mw, mh = (x2 - x1) * margin, (y2 - y1) * margin
    left, top = max(0, int(x1 - mw)), max(0, int(y1 - mh))
    right, bottom = min(img.width, int(x2 + mw) + 1), min(img.height, int(y2 + mh) + 1)
    right, bottom = max(right, min(img.width, left + min_side)), max(bottom, min(img.height, top + min_side))
    return img.crop((left, top, right, bottom))


def jpeg_size(pil_img, quality):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return buf.tell()


def process_image(path, detector, tile_size, overlap):
    """Same chain as pipeline.run_pipeline, with measured payload sizes."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    w, h = img.size
    tiles = generate_tiles(w, h, tile_size=tile_size, overlap_ratio=overlap)
    t0 = time.perf_counter()
    raw = []
    for t in tiles:
        crop = img.crop((t.x_offset, t.y_offset, t.x_offset + t.width, t.y_offset + t.height))
        for d in detector.predict(crop):
            raw.append({"bbox_xyxy": tile_box_to_global(d.bbox_xyxy, t), "confidence": d.confidence})
    det_time = time.perf_counter() - t0

    deduped = nms_merge(raw, iou_threshold=0.4)
    for d in deduped:
        d["tier"] = confidence_tier(d["confidence"])
        crop = crop_with_margin(img, d["bbox_xyxy"])
        small = crop.copy()
        small.thumbnail((64, 64))                                   # low-resolution ROI
        d["bytes_small_roi"] = METADATA_SIZE_BYTES + jpeg_size(small, 60)
        d["bytes_full_crop"] = METADATA_SIZE_BYTES + jpeg_size(crop, 90)   # native-resolution crop
        d["payload_bytes"] = {"high_confidence": METADATA_SIZE_BYTES,
                              "uncertain": d["bytes_small_roi"],
                              "highly_uncertain": d["bytes_full_crop"]}[d["tier"]]
        d["payload_bytes_modeled"] = payload_size_for_detection(d["confidence"])
    return w, h, len(tiles), raw, deduped, det_time


def evaluate_folder(images_dir, labels_dir, detector, out_dir="results",
                    tile_size=640, overlap=0.20, tag=None):
    tag = tag or f"tile{tile_size}"
    files = sorted(f for f in os.listdir(images_dir) if f.lower().endswith(IMG_EXTS))
    if not files:
        raise SystemExit(f"Aucune image trouvée dans {images_dir}")

    rows, det_rows = [], []
    tier_stats = {t: {"detections": 0, "true_positives": 0, "payload_bytes": 0} for t in TIERS}

    for i, f in enumerate(files, 1):
        path = os.path.join(images_dir, f)
        w, h, n_tiles, raw, deduped, det_time = process_image(path, detector, tile_size, overlap)
        label_path = os.path.join(labels_dir, os.path.splitext(f)[0] + ".txt") if labels_dir else None
        gt = load_gt_boxes(label_path, w, h)

        flags = match_flags(deduped, gt) if labels_dir else [False] * len(deduped)
        tp = sum(flags)
        fp, fn = len(deduped) - tp, len(gt) - tp
        lost = (covered_gt(raw, gt) - covered_gt(deduped, gt)) if labels_dir else 0

        for d, ok in zip(deduped, flags):
            ts = tier_stats[d["tier"]]
            ts["detections"] += 1
            ts["true_positives"] += int(ok)
            ts["payload_bytes"] += d["payload_bytes"]
            det_rows.append({"image": f, "confidence": round(d["confidence"], 4), "tier": d["tier"],
                             "true_positive": int(ok), "bytes_small_roi": d["bytes_small_roi"],
                             "bytes_full_crop": d["bytes_full_crop"], "payload_bytes": d["payload_bytes"]})

        rows.append({
            "image": f, "n_ships_truth": len(gt), "tiles": n_tiles,
            "raw_detections": len(raw), "deduped_detections": len(deduped),
            "tp": tp, "fp": fp, "fn": fn, "ships_lost_by_dedup": lost,
            "payload_bytes": sum(d["payload_bytes"] for d in deduped),
            "payload_bytes_modeled": sum(d["payload_bytes_modeled"] for d in deduped),
            "raw_uncompressed_bytes": w * h * 3,
            "jpeg_bytes": os.path.getsize(path),
            "detection_time_s": round(det_time, 4),
        })
        print(f"[{i}/{len(files)}] {f}: truth {len(gt)} | raw {len(raw)} -> dedup {len(deduped)} | "
              f"TP {tp} FP {fp} FN {fn} | lost by dedup {lost}")

    n = len(rows)
    s = lambda k: sum(x[k] for x in rows)
    proposed, raw_unc, jpeg = s("payload_bytes"), s("raw_uncompressed_bytes"), s("jpeg_bytes")
    baseline2 = s("raw_detections") * METADATA_SIZE_BYTES
    metadata_part = s("deduped_detections") * METADATA_SIZE_BYTES
    tp, fp, fn = s("tp"), s("fp"), s("fn")

    summary = {
        "tile_size": tile_size, "overlap": overlap,
        "n_images": n,
        "n_images_with_ships": sum(1 for x in rows if x["n_ships_truth"] > 0),
        "n_ships_truth": tp + fn,
        "bytes": {
            "baseline_1_raw_uncompressed": raw_unc,
            "baseline_1_jpeg_files": jpeg,
            "baseline_2_metadata_per_raw_detection": baseline2,
            "proposed_measured": proposed,
            "proposed_modeled_fixed_sizes": s("payload_bytes_modeled"),
            "proposed_metadata_part": metadata_part,
            "proposed_crop_part": proposed - metadata_part,
        },
        "data_reduction_pct": {
            "vs_raw_uncompressed": round((1 - proposed / raw_unc) * 100, 2),
            "vs_jpeg_files": round((1 - proposed / jpeg) * 100, 2),
            "vs_baseline_2": round((1 - proposed / baseline2) * 100, 2) if baseline2 else None,
        },
        "pipeline_detection_quality": None if not labels_dir else {
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "ships_lost_by_dedup": s("ships_lost_by_dedup"),
            "false_positives_on_empty_images": sum(x["fp"] for x in rows if x["n_ships_truth"] == 0),
        },
        "per_tier": {t: {**v, "precision": round(v["true_positives"] / v["detections"], 3)
                         if v["detections"] else None} for t, v in tier_stats.items()},
        "mean_detection_time_per_image_s": round(s("detection_time_s") / n, 3),
    }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"real_images_results_{tag}.json")
    csv_path = os.path.join(out_dir, f"real_images_per_image_{tag}.csv")
    det_path = os.path.join(out_dir, f"real_images_detections_{tag}.csv")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(csv_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    with open(det_path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["image", "confidence", "tier", "true_positive",
                                            "bytes_small_roi", "bytes_full_crop", "payload_bytes"])
        wr.writeheader()
        wr.writerows(det_rows)

    print("\n" + "=" * 60)
    print(json.dumps(summary, indent=2))
    print(f"\nSauvegardé : {json_path}\n             {csv_path}\n             {det_path}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--weights", default="detection/models/best_v2.pt")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tile-size", type=int, default=640, help="768 = pas de découpage sur des images 768x768")
    ap.add_argument("--overlap", type=float, default=0.20)
    ap.add_argument("--conf", type=float, default=0.25, help="seuil de confiance du détecteur")
    ap.add_argument("--tag", default=None, help="suffixe des fichiers de sortie")
    a = ap.parse_args()

    from detection.detect import ShipDetector
    evaluate_folder(a.images, a.labels, ShipDetector(a.weights, conf_threshold=a.conf),
                    a.out_dir, a.tile_size, a.overlap, a.tag)