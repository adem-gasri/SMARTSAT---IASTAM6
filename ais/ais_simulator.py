"""
ais/ais_simulator.py

Since the team has no access to a real AIS feed for this prototype,
this module generates SYNTHETIC AIS beacon positions correlated with a
fraction of the visual detections, to exercise and evaluate the
matching logic in ais_matcher.py end-to-end.

This is clearly a simulation, not real AIS data, and must be reported
as such in the final report's Limitations section.
"""

import random
from dataclasses import dataclass
from typing import List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temporal.detection_list import Detection  # noqa: E402


@dataclass
class AISBeacon:
    lat: float
    lon: float
    time: float
    mmsi: str  # simulated vessel identifier


def simulate_ais_beacons(detections: List[Detection],
                          confirmation_rate: float = 0.7,
                          position_noise_m: float = 30.0,
                          seed: int = 42) -> List[AISBeacon]:
    """
    For `confirmation_rate` of the given detections, generate a nearby
    AIS beacon (simulating a legitimate, broadcasting vessel). The
    remaining detections get no beacon (simulating either a non-AIS
    vessel or a suspicious unconfirmed contact).
    """
    rng = random.Random(seed)
    beacons = []

    for i, d in enumerate(detections):
        if rng.random() < confirmation_rate:
            # jitter the position slightly to simulate GPS/AIS report noise
            lat_jitter = (rng.random() - 0.5) * (position_noise_m / 111_000)
            lon_jitter = (rng.random() - 0.5) * (position_noise_m / 111_000)
            beacons.append(AISBeacon(
                lat=d.lat + lat_jitter,
                lon=d.lon + lon_jitter,
                time=d.time,
                mmsi=f"SIM{i:06d}",
            ))

    return beacons


if __name__ == "__main__":
    dets = [Detection(lat=36.80 + i * 0.01, lon=10.18 + i * 0.01, confidence=0.7, time=1000.0)
            for i in range(10)]
    beacons = simulate_ais_beacons(dets, confirmation_rate=0.7)
    print(f"{len(dets)} detections -> {len(beacons)} simulated AIS beacons generated")