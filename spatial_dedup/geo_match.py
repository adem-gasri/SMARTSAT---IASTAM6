"""
spatial_dedup/geo_match.py

Across neighbouring images, a ship near an image boundary may be
independently detected in more than one capture. This module compares
detections using geographic distance (Equation 2 in the paper):

    d(Di, Dj) < Ts

where Ts is a spatial matching threshold (in metres). Distance alone
does not guarantee identity, so this module also checks that the
observation times are close enough to be physically plausible for the
same ship.

Matches the paper's Section III-E (second redundancy axis).
"""

import math
from typing import List, Dict, Any, Tuple


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two lat/lon points."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def geo_match(detections_a: List[Dict[str, Any]],
              detections_b: List[Dict[str, Any]],
              spatial_threshold_m: float = 150.0,
              max_time_delta_s: float = 300.0) -> List[Tuple[Dict, Dict]]:
    """
    detections_a / detections_b: lists of dicts with at least
        {"lat": float, "lon": float, "time": float (unix seconds), ...}
    Returns list of (det_a, det_b) pairs considered to be the same
    physical ship observed across the two neighbouring images.
    """
    matches = []
    used_b = set()

    for da in detections_a:
        best_match = None
        best_dist = spatial_threshold_m
        for idx, db in enumerate(detections_b):
            if idx in used_b:
                continue
            if abs(da["time"] - db["time"]) > max_time_delta_s:
                continue
            dist = haversine_m(da["lat"], da["lon"], db["lat"], db["lon"])
            if dist < best_dist:
                best_dist = dist
                best_match = (idx, db)

        if best_match is not None:
            idx, db = best_match
            used_b.add(idx)
            matches.append((da, db))

    return matches


def deduplicate_cross_image(detections_a: List[Dict[str, Any]],
                             detections_b: List[Dict[str, Any]],
                             spatial_threshold_m: float = 150.0) -> List[Dict[str, Any]]:
    """Returns a single merged list with cross-image duplicates removed,
    keeping the higher-confidence detection of each matched pair."""
    matches = geo_match(detections_a, detections_b, spatial_threshold_m)
    matched_a_ids = {id(da) for da, _ in matches}
    matched_b_ids = {id(db) for _, db in matches}

    merged = [d for d in detections_a if id(d) not in matched_a_ids]
    merged += [d for d in detections_b if id(d) not in matched_b_ids]
    for da, db in matches:
        merged.append(da if da["confidence"] >= db["confidence"] else db)

    return merged


if __name__ == "__main__":
    img_a = [{"lat": 36.8065, "lon": 10.1815, "time": 1000.0, "confidence": 0.77}]
    img_b = [{"lat": 36.8066, "lon": 10.1816, "time": 1005.0, "confidence": 0.82},   # same ship
             {"lat": 36.9000, "lon": 10.3000, "time": 1005.0, "confidence": 0.55}]   # distinct ship

    merged = deduplicate_cross_image(img_a, img_b)
    print(f"{len(img_a) + len(img_b)} raw detections across 2 images -> {len(merged)} after geo dedup")
    for d in merged:
        print(d)