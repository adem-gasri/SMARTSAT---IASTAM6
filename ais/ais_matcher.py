"""
ais/ais_matcher.py

Cross-references deduplicated visual detections against AIS beacon
positions (Section III-H in the paper). A beacon is associated with a
visual detection when their positions fall within a fixed error margin
epsilon. A detection with no matching beacon is flagged "unconfirmed".
"""

import math
from typing import List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temporal.detection_list import Detection  # noqa: E402
from ais.ais_simulator import AISBeacon  # noqa: E402


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def match_ais(detections: List[Detection], beacons: List[AISBeacon],
              epsilon_m: float = 100.0) -> List[Detection]:
    """
    Mutates and returns the detections list with `.ais_status` set to
    "confirmed" or "unconfirmed" for each detection.
    """
    for d in detections:
        d.ais_status = "unconfirmed"
        for b in beacons:
            if haversine_m(d.lat, d.lon, b.lat, b.lon) < epsilon_m:
                d.ais_status = "confirmed"
                break
    return detections


if __name__ == "__main__":
    from ais.ais_simulator import simulate_ais_beacons

    dets = [Detection(lat=36.80 + i * 0.01, lon=10.18 + i * 0.01, confidence=0.7, time=1000.0)
            for i in range(10)]
    beacons = simulate_ais_beacons(dets, confirmation_rate=0.7)
    matched = match_ais(dets, beacons, epsilon_m=100.0)

    confirmed = sum(1 for d in matched if d.ais_status == "confirmed")
    print(f"{confirmed}/{len(matched)} detections confirmed by AIS")