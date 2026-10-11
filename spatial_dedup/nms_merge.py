"""
spatial_dedup/nms_merge.py

Within a single capture, overlapping tiles may each detect the same
physical ship. This module merges per-tile boxes (already converted to
global image coordinates, see tiling.py) using Non-Maximum Suppression:
whenever two boxes' IoU exceeds a threshold (0.4 by default), only the
highest-confidence detection is kept.

Matches the paper's Section III-E (first redundancy axis).
"""

from typing import List, Dict, Any


def iou(box_a: List[float], box_b: List[float]) -> float:
    """Standard Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def nms_merge(detections: List[Dict[str, Any]], iou_threshold: float = 0.4) -> List[Dict[str, Any]]:
    """
    detections: list of dicts, each with at least:
        {"bbox_xyxy": [x1,y1,x2,y2], "confidence": float, ...}
    Returns a filtered list with duplicates removed, keeping the
    highest-confidence detection in each overlapping group.
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    suppressed = [False] * len(sorted_dets)

    for i, det_i in enumerate(sorted_dets):
        if suppressed[i]:
            continue
        kept.append(det_i)
        for j in range(i + 1, len(sorted_dets)):
            if suppressed[j]:
                continue
            if iou(det_i["bbox_xyxy"], sorted_dets[j]["bbox_xyxy"]) > iou_threshold:
                suppressed[j] = True

    return kept


if __name__ == "__main__":
    # quick self-test: two overlapping tile detections of the same ship
    dets = [
        {"bbox_xyxy": [100, 100, 160, 140], "confidence": 0.81},
        {"bbox_xyxy": [108, 104, 166, 142], "confidence": 0.74},  # same ship, other tile
        {"bbox_xyxy": [400, 400, 460, 440], "confidence": 0.60},  # distinct ship
    ]
    merged = nms_merge(dets, iou_threshold=0.4)
    print(f"{len(dets)} raw detections -> {len(merged)} after NMS merge")
    for d in merged:
        print(d)