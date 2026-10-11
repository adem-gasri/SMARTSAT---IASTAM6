"""
temporal/detection_list.py

Simple data structure for a "detection list" L_k at observation time
t_k, as formalised in the paper (Section III-G, Equation 1):

    D_i = (lat_i, lon_i, c_i, t_i, type_i)
    L_k = {D_1, ..., D_n}
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Detection:
    lat: float
    lon: float
    confidence: float
    time: float          # unix seconds
    type: str = "ship"
    ais_status: str = "unknown"   # "confirmed" | "unconfirmed" | "unknown"


@dataclass
class DetectionList:
    observation_time: float
    detections: List[Detection] = field(default_factory=list)

    def __len__(self):
        return len(self.detections)