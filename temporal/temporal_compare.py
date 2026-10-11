"""
temporal/temporal_compare.py

The satellite may observe the same ocean region repeatedly. This
module compares the current detection list L_k with the previous list
L_{k-1}:

    if d(D, D') < Tm      -> position unchanged, nothing transmitted
    if d(D, D') >= Tm     -> meaningful displacement, transmit update
    if no match found     -> new observation, transmit

Matches the paper's Section III-G (temporal redundancy reduction).
"""

import math
from typing import List, Optional, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temporal.detection_list import Detection, DetectionList  # noqa: E402


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def find_closest(detection: Detection, previous: List[Detection],
                  max_search_radius_m: float = 2000.0) -> Optional[Detection]:
    best = None
    best_dist = max_search_radius_m
    for prev in previous:
        dist = haversine_m(detection.lat, detection.lon, prev.lat, prev.lon)
        if dist < best_dist:
            best_dist = dist
            best = prev
    return best


def compare_lists(current: DetectionList, previous: Optional[DetectionList],
                   movement_threshold_m: float = 50.0) -> Tuple[List[Detection], List[Detection]]:
    """
    Returns (to_transmit, suppressed):
      - to_transmit: detections that are new or meaningfully displaced
      - suppressed: detections considered unchanged since last observation
    """
    if previous is None or len(previous) == 0:
        return list(current.detections), []

    to_transmit, suppressed = [], []

    for d in current.detections:
        match = find_closest(d, previous.detections)
        if match is None:
            to_transmit.append(d)  # new observation
            continue
        dist = haversine_m(d.lat, d.lon, match.lat, match.lon)
        if dist >= movement_threshold_m:
            to_transmit.append(d)  # meaningful displacement
        else:
            suppressed.append(d)   # unchanged -> no retransmission

    return to_transmit, suppressed


if __name__ == "__main__":
    prev = DetectionList(observation_time=1000.0, detections=[
        Detection(lat=36.8065, lon=10.1815, confidence=0.80, time=1000.0),  # stationary ship
    ])
    curr = DetectionList(observation_time=1090.0, detections=[
        Detection(lat=36.8065, lon=10.18151, confidence=0.83, time=1090.0),  # same ship, hasn't moved
        Detection(lat=36.8200, lon=10.2000, confidence=0.70, time=1090.0),   # new ship
    ])

    to_tx, suppressed = compare_lists(curr, prev, movement_threshold_m=50.0)
    print(f"To transmit: {len(to_tx)} | Suppressed (unchanged): {len(suppressed)}")
    for d in to_tx:
        print("  TX:", d)
    for d in suppressed:
        print("  SUPPRESSED:", d)