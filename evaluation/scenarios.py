"""
evaluation/scenarios.py

Builds a synthetic but structured test set covering the key scenarios
listed in the paper (Section IV-D), so the pipeline can be evaluated
end-to-end without waiting on a full re-run of the real detector on
the complete Airbus dataset.

Replace this with actual YOLOv8n outputs (via detection/detect.py)
once available; the rest of the evaluation pipeline is agnostic to
where detections come from.

NOTE: This uses a fixed random seed so results are reproducible and
comparable run to run, which matters for a final report.
"""

import random
from typing import List, Dict, Any
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temporal.detection_list import Detection, DetectionList  # noqa: E402


def build_synthetic_scenario(seed: int = 7) -> Dict[str, Any]:
    """
    Builds a small multi-observation scenario:
      - 1 stationary ship observed 3 times (same position, small noise)
      - 1 moving ship observed 3 times (position drifts each time)
      - 1 new ship appearing only in the last observation
      - 1 pair of detections of the same ship in two overlapping tiles
        (for spatial dedup testing), plus one distinct ship
    """
    rng = random.Random(seed)
    base_time = 1_700_000_000.0
    orbit_gap_s = 90 * 60  # one orbit between observations

    stationary_lat, stationary_lon = 36.8065, 10.1815
    moving_lat, moving_lon = 37.0000, 10.5000

    observations: List[DetectionList] = []

    for k in range(3):
        t = base_time + k * orbit_gap_s
        dets = []

        # stationary ship: tiny GPS-noise-level jitter only
        dets.append(Detection(
            lat=stationary_lat + rng.uniform(-0.00002, 0.00002),
            lon=stationary_lon + rng.uniform(-0.00002, 0.00002),
            confidence=rng.uniform(0.78, 0.9),
            time=t,
        ))

        # moving ship: drifts ~0.01 deg (~1.1 km) between observations
        drift = k * 0.01
        dets.append(Detection(
            lat=moving_lat + drift,
            lon=moving_lon + drift,
            confidence=rng.uniform(0.6, 0.75),
            time=t,
        ))

        # new ship appears only on the last pass
        if k == 2:
            dets.append(Detection(
                lat=36.95, lon=10.40,
                confidence=rng.uniform(0.5, 0.65),
                time=t,
            ))

        observations.append(DetectionList(observation_time=t, detections=dets))

    # Boundary-pair raw detections (pre-NMS): same physical ship seen in
    # two overlapping tiles with slightly different pixel coordinates.
    boundary_pair_raw_detections = [
        {"bbox_xyxy": [612, 300, 660, 335], "confidence": 0.79},
        {"bbox_xyxy": [620, 304, 668, 338], "confidence": 0.73},  # same ship, neighbouring tile
        {"bbox_xyxy": [50, 50, 90, 80], "confidence": 0.58},      # distinct ship, no overlap
    ]

    return {
        "num_raw_images": 45,  # 40 empty + 5 images actually containing the above ships
        "observations": observations,
        "boundary_pair_raw_detections": boundary_pair_raw_detections,
        # ground truth for error-rate metrics (hand-labelled for this synthetic set)
        "ground_truth": {
            "total_merges_expected": 1,       # the boundary pair should merge into 1
            "incorrect_merges": 0,            # NMS correctly avoided merging the distinct ship
            "total_meaningful_movements": 1,  # the moving ship's final hop is meaningful
            "ships_with_no_ais": 1,           # one ship intentionally has no beacon
        },
    }


if __name__ == "__main__":
    scenario = build_synthetic_scenario()
    print(f"Raw images in scenario: {scenario['num_raw_images']}")
    for i, obs in enumerate(scenario["observations"]):
        print(f"  Observation {i}: {len(obs)} detections at t={obs.observation_time}")
    print(f"Boundary-pair raw detections: {len(scenario['boundary_pair_raw_detections'])}")