
import os
import csv
import json
import time
import argparse
from pathlib import Path

from ultralytics import YOLO


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

METADATA_BYTES = 64
SMALL_ROI_BYTES = 8 * 1024
FULL_CROP_BYTES = 60 * 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="SmartSat real-image evaluation"
    )

    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--tag", default="real_images")
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--high-thr", type=float, default=0.75)
    parser.add_argument("--low-thr", type=float, default=0.45)

    return parser.parse_args()


def read_labels(label_path, image_width, image_height):
    """Read ground-truth boxes in YOLO normalized format."""
    boxes = []

    if not os.path.isfile(label_path):
        return boxes

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            try:
                xc, yc, width, height = map(float, parts[1:5])
            except ValueError:
                continue

            x1 = (xc - width / 2) * image_width
            y1 = (yc - height / 2) * image_height
            x2 = (xc + width / 2) * image_width
            y2 = (yc + height / 2) * image_height

            boxes.append([x1, y1, x2, y2])

    return boxes


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def get_tier(confidence, high_thr, low_thr):
    if confidence >= high_thr:
        return "high"
    elif confidence >= low_thr:
        return "uncertain"
    else:
        return "highly_uncertain"


def calculate_payload(tier):
    if tier == "high":
        return METADATA_BYTES, 0, 0

    elif tier == "uncertain":
        return METADATA_BYTES + SMALL_ROI_BYTES, SMALL_ROI_BYTES, 0

    else:
        return METADATA_BYTES + FULL_CROP_BYTES, 0, FULL_CROP_BYTES


def match_detections(predictions, ground_truth, iou_threshold):
    """Match predictions to ground truth, with each GT used once."""
    used_gt = set()
    tp = 0
    fp = 0

    predictions.sort(
        key=lambda prediction: prediction["confidence"],
        reverse=True
    )

    for prediction in predictions:
        best_iou = 0
        best_index = None

        for index, gt_box in enumerate(ground_truth):
            if index in used_gt:
                continue

            score = calculate_iou(
                prediction["bbox_xyxy"],
                gt_box
            )

            if score > best_iou:
                best_iou = score
                best_index = index

        if best_index is not None and best_iou >= iou_threshold:
            tp += 1
            used_gt.add(best_index)
            prediction["true_positive"] = True
        else:
            fp += 1
            prediction["true_positive"] = False

    fn = len(ground_truth) - len(used_gt)

    return tp, fp, fn


def safe_divide(a, b):
    return a / b if b != 0 else 0.0


def main():
    args = parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    weights_path = Path(args.weights)

    if not images_dir.is_dir():
        raise SystemExit(f"Images folder not found: {images_dir}")

    if not labels_dir.is_dir():
        raise SystemExit(f"Labels folder not found: {labels_dir}")

    if not weights_path.is_file():
        raise SystemExit(f"Model not found: {weights_path}")

    image_paths = sorted(
        path for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading YOLO model...")
    model = YOLO(str(weights_path))

    det_rows = []
    image_rows = []
    gt_by_image = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_jpeg_bytes = 0
    total_raw_bytes = 0
    total_payload_bytes = 0
    total_inference_time = 0.0

    for number, image_path in enumerate(image_paths, start=1):
        print(f"[{number}/{len(image_paths)}] {image_path.name}")

        start = time.perf_counter()

        results = model.predict(
            source=str(image_path),
            imgsz=args.tile_size,
            conf=args.conf,
            verbose=False
        )

        inference_time = time.perf_counter() - start
        total_inference_time += inference_time

        result = results[0]
        image_height, image_width = result.orig_shape

        label_path = labels_dir / f"{image_path.stem}.txt"

        gt_boxes = read_labels(
            str(label_path),
            image_width,
            image_height
        )

        gt_by_image[image_path.name] = gt_boxes

        predictions = []

        if result.boxes is not None:
            for box in result.boxes:
                coordinates = [
                    float(value)
                    for value in box.xyxy[0].tolist()
                ]

                confidence = float(box.conf[0].item())

                predictions.append({
                    "image": image_path.name,
                    "confidence": confidence,
                    "bbox_xyxy": coordinates
                })

        tp, fp, fn = match_detections(
            predictions,
            gt_boxes,
            args.iou
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        jpeg_bytes = image_path.stat().st_size
        raw_bytes = image_width * image_height * 3

        total_jpeg_bytes += jpeg_bytes
        total_raw_bytes += raw_bytes

        image_payload = 0

        for prediction in predictions:
            confidence = prediction["confidence"]

            tier = get_tier(
                confidence,
                args.high_thr,
                args.low_thr
            )

            payload, small_roi, full_crop = calculate_payload(tier)
            image_payload += payload

            x1, y1, x2, y2 = prediction["bbox_xyxy"]

            det_rows.append({
                "image": image_path.name,
                "confidence": round(confidence, 6),
                "tier": tier,
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "true_positive": prediction["true_positive"],
                "bytes_small_roi": small_roi,
                "bytes_full_crop": full_crop,
                "payload_bytes": payload
            })

        total_payload_bytes += image_payload

        image_rows.append({
            "image": image_path.name,
            "image_bytes_jpeg": jpeg_bytes,
            "image_bytes_raw_rgb": raw_bytes,
            "ground_truth": len(gt_boxes),
            "detections": len(predictions),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": round(safe_divide(tp, tp + fp), 6),
            "recall": round(safe_divide(tp, tp + fn), 6),
            "payload_bytes_estimated": image_payload,
            "inference_seconds": round(inference_time, 6)
        })

    precision = safe_divide(total_tp, total_tp + total_fp)
    recall = safe_divide(total_tp, total_tp + total_fn)
    f1 = safe_divide(
        2 * precision * recall,
        precision + recall
    )

    summary = {
        "tag": args.tag,
        "num_images": len(image_paths),
        "tile_size": args.tile_size,
        "overlap_argument": args.overlap,
        "confidence_threshold": args.conf,
        "iou_threshold": args.iou,
        "TP": total_tp,
        "FP": total_fp,
        "FN": total_fn,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1_score": round(f1, 6),
        "mean_inference_seconds_per_image": round(
            total_inference_time / len(image_paths), 6
        ),
        "total_jpeg_bytes": total_jpeg_bytes,
        "total_raw_rgb_bytes": total_raw_bytes,
        "estimated_payload_bytes": total_payload_bytes,
        "reduction_vs_jpeg_percent": round(
            (1 - total_payload_bytes / total_jpeg_bytes) * 100, 4
        ) if total_jpeg_bytes else 0,
        "reduction_vs_raw_percent": round(
            (1 - total_payload_bytes / total_raw_bytes) * 100, 4
        ) if total_raw_bytes else 0,
        "payload_note": (
            "Estimated payload: 64-byte metadata, 8-KiB ROI, "
            "60-KiB crop. Not actual network traffic."
        )
    }

    detections_csv = output_dir / f"real_images_detections_{args.tag}.csv"
    per_image_csv = output_dir / f"real_images_per_image_{args.tag}.csv"
    summary_json = output_dir / f"real_images_summary_{args.tag}.json"
    ground_truth_json = output_dir / f"real_images_ground_truth_{args.tag}.json"

    detection_fields = [
        "image", "confidence", "tier",
        "x1", "y1", "x2", "y2",
        "true_positive",
        "bytes_small_roi", "bytes_full_crop", "payload_bytes"
    ]

    with open(detections_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=detection_fields)
        writer.writeheader()
        writer.writerows(det_rows)

    image_fields = [
        "image", "image_bytes_jpeg", "image_bytes_raw_rgb",
        "ground_truth", "detections", "TP", "FP", "FN",
        "precision", "recall", "payload_bytes_estimated",
        "inference_seconds"
    ]

    with open(per_image_csv, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=image_fields)
        writer.writeheader()
        writer.writerows(image_rows)

    with open(summary_json, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    with open(ground_truth_json, "w", encoding="utf-8") as file:
        json.dump({
            "tile_size": args.tile_size,
            "iou_threshold": args.iou,
            "ground_truth": gt_by_image
        }, file, indent=2, ensure_ascii=False)

    print("\n========== SMARTSAT RESULTS ==========")
    print(f"Images: {len(image_paths)}")
    print(f"TP: {total_tp}")
    print(f"FP: {total_fp}")
    print(f"FN: {total_fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(
        "Mean inference time/image: "
        f"{total_inference_time / len(image_paths):.4f} s"
    )
    print(f"JPEG bytes: {total_jpeg_bytes:,}")
    print(f"Raw RGB bytes: {total_raw_bytes:,}")
    print(f"Estimated payload: {total_payload_bytes:,}")
    print(
        "Reduction vs JPEG: "
        f"{summary['reduction_vs_jpeg_percent']:.2f}%"
    )
    print(
        "Reduction vs raw: "
        f"{summary['reduction_vs_raw_percent']:.2f}%"
    )

    print("\nFiles saved:")
    print(detections_csv)
    print(per_image_csv)
    print(summary_json)
    print(ground_truth_json)


if __name__ == "__main__":
    main()
