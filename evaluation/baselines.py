"""
evaluation/baselines.py

Implements the two baselines defined in the paper (Section IV-B) so
the proposed system can be compared against them on the same test set.

  Baseline 1 - Raw-Image Transmission:
      every captured image is transmitted in full, no onboard processing.

  Baseline 2 - Detection-Based Transmission:
      ships are detected onboard and metadata is transmitted for every
      detection, WITHOUT spatial, temporal, or AIS-based redundancy
      reduction, and without confidence-aware payload sizing.

  Proposed System:
      onboard detection + confidence-aware transmission + spatial
      dedup + temporal comparison + AIS matching, combined.
"""

from typing import List, Dict, Any

# ---- Illustrative payload sizes (bytes), consistent with the paper ----
RAW_IMAGE_SIZE_BYTES = 768 * 768 * 3          # uncompressed 768x768 RGB tile
METADATA_SIZE_BYTES = 64                       # lat, lon, type, confidence, time
SMALL_ROI_SIZE_BYTES = 8 * 1024                # ~8 KB low-res crop
FULL_CROP_SIZE_BYTES = 60 * 1024               # ~60 KB high-res crop / full image


def baseline_1_raw(num_images: int) -> int:
    """Every image transmitted in full, regardless of content."""
    return num_images * RAW_IMAGE_SIZE_BYTES


def baseline_2_detection_only(all_detections: List[Dict[str, Any]]) -> int:
    """Metadata transmitted for EVERY detection, no dedup, no AIS, no
    confidence-aware sizing."""
    return len(all_detections) * METADATA_SIZE_BYTES


def confidence_tier(confidence: float, high_thr: float = 0.75, low_thr: float = 0.45) -> str:
    if confidence >= high_thr:
        return "high_confidence"
    elif confidence >= low_thr:
        return "uncertain"
    return "highly_uncertain"


def payload_size_for_detection(confidence: float) -> int:
    tier = confidence_tier(confidence)
    if tier == "high_confidence":
        return METADATA_SIZE_BYTES
    elif tier == "uncertain":
        return METADATA_SIZE_BYTES + SMALL_ROI_SIZE_BYTES
    return METADATA_SIZE_BYTES + FULL_CROP_SIZE_BYTES


def proposed_system(final_detections_to_transmit: List[Dict[str, Any]]) -> int:
    """
    final_detections_to_transmit: detections that survived spatial
    dedup + temporal comparison (i.e. actually need to be sent), each
    with a "confidence" field. Payload size adapts to confidence tier.
    """
    return sum(payload_size_for_detection(d["confidence"]) for d in final_detections_to_transmit)


if __name__ == "__main__":
    num_images = 100
    all_detections = [{"confidence": 0.8} for _ in range(40)] + [{"confidence": 0.3} for _ in range(10)]
    final_to_tx = [{"confidence": 0.8} for _ in range(12)] + [{"confidence": 0.55} for _ in range(5)]

    b1 = baseline_1_raw(num_images)
    b2 = baseline_2_detection_only(all_detections)
    prop = proposed_system(final_to_tx)

    print(f"Baseline 1 (raw):        {b1/1024:.1f} KB")
    print(f"Baseline 2 (detection):  {b2/1024:.1f} KB")
    print(f"Proposed system:         {prop/1024:.1f} KB")
    print(f"Reduction vs Baseline 1: {(1 - prop/b1)*100:.1f}%")
    print(f"Reduction vs Baseline 2: {(1 - prop/b2)*100:.1f}%")