"""
pipeline.py

End-to-end SmartSat pipeline on a REAL image, using the trained detector:

    Image -> Tiling -> Detection per tile -> Spatial dedup (NMS)
    -> Confidence-aware payload sizing -> saved outputs

Outputs (written to --out-dir, default "results"):
    pipeline_output.json   what would be transmitted, with sizes and timings
    pipeline_output.jpg    the image with detections drawn (colour = tier)

Usage:
    python pipeline.py --image detection/results/detection_example.jpg --show-tiles
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spatial_dedup.tiling import generate_tiles, tile_box_to_global      # noqa: E402
from spatial_dedup.nms_merge import nms_merge                            # noqa: E402
from evaluation.baselines import (                                       # noqa: E402
    payload_size_for_detection, confidence_tier, RAW_IMAGE_SIZE_BYTES,
)
from evaluation.metrics import timed, data_reduction_rate                # noqa: E402

TIER_COLORS = {
    "high_confidence": (46, 204, 113),    # green  : metadata only
    "uncertain": (243, 156, 18),          # amber  : metadata + small crop
    "highly_uncertain": (231, 76, 60),    # red    : metadata + full crop
}


def run_pipeline(image_path, weights_path, tile_size=640, overlap_ratio=0.20, detector=None):
    from PIL import Image

    if detector is None:
        from detection.detect import ShipDetector
        detector = ShipDetector(weights_path)

    stage_times = {}
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    with timed("tiling", stage_times):
        tiles = generate_tiles(width, height, tile_size=tile_size, overlap_ratio=overlap_ratio)

    with timed("detection", stage_times):
        all_global_detections = []
        for tile in tiles:
            crop = img.crop((tile.x_offset, tile.y_offset,
                             tile.x_offset + tile.width, tile.y_offset + tile.height))
            for d in detector.predict(crop):
                all_global_detections.append({
                    "bbox_xyxy": tile_box_to_global(d.bbox_xyxy, tile),
                    "confidence": d.confidence,
                    "class_name": d.class_name,
                })

    with timed("spatial_dedup", stage_times):
        deduped = nms_merge(all_global_detections, iou_threshold=0.4)

    with timed("payload_sizing", stage_times):
        for d in deduped:
            d["tier"] = confidence_tier(d["confidence"])
            d["payload_bytes"] = payload_size_for_detection(d["confidence"])
        payload_bytes = sum(d["payload_bytes"] for d in deduped)

    raw_bytes = RAW_IMAGE_SIZE_BYTES
    return {
        "image": os.path.basename(image_path),
        "image_size": [width, height],
        "num_tiles": len(tiles),
        "tiles": [[t.x_offset, t.y_offset, t.width, t.height] for t in tiles],
        "raw_detections": len(all_global_detections),
        "deduped_detections": len(deduped),
        "payload_bytes": payload_bytes,
        "raw_image_bytes": raw_bytes,
        "data_reduction_pct": round(data_reduction_rate(raw_bytes, payload_bytes) * 100, 2),
        "stage_times_s": {k: round(v, 6) for k, v in stage_times.items()},
        "detections": [
            {"bbox_xyxy": [round(v, 1) for v in d["bbox_xyxy"]],
             "confidence": round(d["confidence"], 3),
             "tier": d["tier"], "payload_bytes": d["payload_bytes"]}
            for d in deduped
        ],
    }


def save_outputs(result, image_path, out_dir="results", show_tiles=False):
    from PIL import Image, ImageDraw

    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "pipeline_output.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    if show_tiles:
        for x, y, w, h in result["tiles"]:
            draw.rectangle([x, y, x + w - 1, y + h - 1], outline=(120, 170, 255), width=1)

    for d in result["detections"]:
        color = TIER_COLORS[d["tier"]]
        x1, y1, x2, y2 = d["bbox_xyxy"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"ship {d['confidence']:.2f} | {d['payload_bytes']} B"
        ty = max(0, y1 - 12)
        draw.rectangle([x1, ty, x1 + 7 * len(label), ty + 12], fill=color)
        draw.text((x1 + 2, ty), label, fill=(0, 0, 0))

    img_path = os.path.join(out_dir, "pipeline_output.jpg")
    img.save(img_path, quality=92)
    return json_path, img_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartSat end-to-end pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", default="detection/models/best.pt")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.20)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--show-tiles", action="store_true", help="draw the tile grid on the output image")
    args = parser.parse_args()

    r = run_pipeline(args.image, args.weights, args.tile_size, args.overlap)
    json_path, img_path = save_outputs(r, args.image, args.out_dir, args.show_tiles)

    print(f"Image size: {r['image_size']}  |  tiles: {r['num_tiles']}")
    print(f"Raw detections: {r['raw_detections']} -> after dedup: {r['deduped_detections']}")
    print(f"Payload: {r['payload_bytes']} bytes vs raw image {r['raw_image_bytes']} bytes")
    print(f"Data reduction: {r['data_reduction_pct']}%")
    print(f"Stage times (s): {r['stage_times_s']}")
    for d in r["detections"]:
        print(f"  ship {d['bbox_xyxy']} conf {d['confidence']} tier {d['tier']} -> {d['payload_bytes']} B")
    print(f"\nSaved:\n  {json_path}\n  {img_path}")