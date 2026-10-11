"""
detection/detect.py

Thin wrapper around the trained YOLOv8n ship detector.
Converts raw Ultralytics results into plain objects that the rest
of the pipeline (spatial dedup, temporal comparison, AIS matching) can
consume without depending on Ultralytics internals.

Usage:
    from detection.detect import ShipDetector

    detector = ShipDetector("detection/models/best.pt")
    detections = detector.predict("path/to/image.jpg")   # path OR PIL image
"""

from dataclasses import dataclass
from typing import List


@dataclass
class RawDetection:
    """One detection in image-pixel coordinates, before geolocation."""
    bbox_xyxy: List[float]      # [x1, y1, x2, y2] in pixels
    confidence: float
    class_name: str = "ship"


class ShipDetector:
    """Loads the trained YOLOv8n model and exposes a simple .predict()."""

    def __init__(self, weights_path: str, conf_threshold: float = 0.25):
        from ultralytics import YOLO  # lazy import: rest of pipeline works without it
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold

    def predict(self, source) -> List[RawDetection]:
        """`source` can be an image path or a PIL.Image (e.g. one tile)."""
        results = self.model.predict(source, conf=self.conf_threshold, verbose=False)
        detections = []
        for box in results[0].boxes:
            detections.append(
                RawDetection(
                    bbox_xyxy=box.xyxy.tolist()[0],
                    confidence=float(box.conf),
                    class_name=self.model.names[int(box.cls)],
                )
            )
        return detections