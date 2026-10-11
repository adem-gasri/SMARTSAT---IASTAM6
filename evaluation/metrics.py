"""
evaluation/metrics.py

Computes every metric listed in Table I of the paper:
  - Data reduction rate
  - Precision / Recall / F1 / mAP (passed in from YOLO's model.val())
  - False merging rate      (spatial dedup errors)
  - Missed update rate      (temporal dedup errors)
  - AIS mismatch rate       (AIS matching errors)
  - Latency                 (measured wall-clock time per stage)
  - Computational cost      (measured wall-clock time, proxy for FLOPs)
"""

import time
from typing import List, Dict, Any, Callable
from contextlib import contextmanager


def data_reduction_rate(d_raw_bytes: int, d_tx_bytes: int) -> float:
    """Equation 6 in the paper: DR = 1 - (D_tx / D_raw)."""
    if d_raw_bytes == 0:
        return 0.0
    return 1 - (d_tx_bytes / d_raw_bytes)


def false_merging_rate(num_incorrect_merges: int, num_total_merges: int) -> float:
    """Fraction of NMS/geo-match merges that incorrectly fused two
    distinct ships into one detection."""
    if num_total_merges == 0:
        return 0.0
    return num_incorrect_merges / num_total_merges


def missed_update_rate(num_suppressed_but_meaningful: int, num_total_meaningful_movements: int) -> float:
    """Fraction of real, meaningful ship movements that the temporal
    comparison incorrectly suppressed (treated as 'unchanged')."""
    if num_total_meaningful_movements == 0:
        return 0.0
    return num_suppressed_but_meaningful / num_total_meaningful_movements


def ais_mismatch_rate(num_incorrect_ais_decisions: int, num_total_ais_decisions: int) -> float:
    """Fraction of AIS confirm/flag decisions that were wrong (false
    confirmation or false 'unconfirmed' flag), against the simulated
    ground truth."""
    if num_total_ais_decisions == 0:
        return 0.0
    return num_incorrect_ais_decisions / num_total_ais_decisions


@contextmanager
def timed(label: str, store: Dict[str, float]):
    """Context manager that records wall-clock duration of a pipeline
    stage into `store[label]` (seconds)."""
    start = time.perf_counter()
    yield
    store[label] = time.perf_counter() - start


def summarize_latency(stage_times_s: Dict[str, float]) -> Dict[str, float]:
    total = sum(stage_times_s.values())
    return {**stage_times_s, "total_latency_s": total}


if __name__ == "__main__":
    # quick self-test
    dr = data_reduction_rate(d_raw_bytes=172_800_000, d_tx_bytes=42_000)
    print(f"Data reduction rate: {dr*100:.2f}%")

    fmr = false_merging_rate(num_incorrect_merges=2, num_total_merges=40)
    print(f"False merging rate: {fmr*100:.2f}%")

    mur = missed_update_rate(num_suppressed_but_meaningful=1, num_total_meaningful_movements=20)
    print(f"Missed update rate: {mur*100:.2f}%")

    amr = ais_mismatch_rate(num_incorrect_ais_decisions=3, num_total_ais_decisions=50)
    print(f"AIS mismatch rate: {amr*100:.2f}%")

    stage_times = {}
    with timed("detection", stage_times):
        time.sleep(0.01)
    with timed("dedup", stage_times):
        time.sleep(0.005)
    print(summarize_latency(stage_times))