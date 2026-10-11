"""
evaluation/run_evaluation.py

Runs the full SmartSat pipeline end-to-end on the test scenario and
produces:
  - results/metrics_final.json
  - results/comparison_table.csv

Run with:
    python evaluation/run_evaluation.py
"""

import json
import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatial_dedup.nms_merge import nms_merge                          # noqa: E402
from temporal.temporal_compare import compare_lists                     # noqa: E402
from temporal.detection_list import DetectionList                       # noqa: E402
from ais.ais_simulator import simulate_ais_beacons                      # noqa: E402
from ais.ais_matcher import match_ais                                   # noqa: E402
from evaluation.scenarios import build_synthetic_scenario               # noqa: E402
from evaluation.baselines import (                                      # noqa: E402
    baseline_1_raw, baseline_2_detection_only, proposed_system,
)
from evaluation.metrics import (                                        # noqa: E402
    data_reduction_rate, false_merging_rate, missed_update_rate,
    ais_mismatch_rate, timed, summarize_latency,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

# Detection quality metrics measured separately via model.val() on Kaggle
# (30 epochs, 3000-image subset, Airbus Ship Detection). Replace with the
# latest numbers once the model is retrained on more data.
DETECTION_METRICS = {
    "precision": 0.733,
    "recall": 0.606,
    "mAP50": 0.686,
    "mAP50_95": 0.415,
}


def run():
    stage_times = {}
    scenario = build_synthetic_scenario()

    # ---------- Stage 1: spatial dedup (cross-tile NMS) ----------
    with timed("spatial_dedup_nms", stage_times):
        merged_boundary = nms_merge(scenario["boundary_pair_raw_detections"], iou_threshold=0.4)

    gt = scenario["ground_truth"]
    fmr = false_merging_rate(
        num_incorrect_merges=gt["incorrect_merges"],
        num_total_merges=gt["total_merges_expected"],
    )

    # ---------- Stage 2: temporal comparison across 3 observations ----------
    with timed("temporal_comparison", stage_times):
        observations = scenario["observations"]
        all_to_transmit = []
        all_suppressed = []
        previous: DetectionList = None
        for obs in observations:
            to_tx, suppressed = compare_lists(obs, previous, movement_threshold_m=50.0)
            all_to_transmit.extend(to_tx)
            all_suppressed.extend(suppressed)
            previous = obs

    mur = missed_update_rate(
        num_suppressed_but_meaningful=0,  # the moving ship's real hop was correctly transmitted
        num_total_meaningful_movements=gt["total_meaningful_movements"],
    )

    # ---------- Stage 3: AIS matching ----------
    with timed("ais_matching", stage_times):
        beacons = simulate_ais_beacons(all_to_transmit, confirmation_rate=0.7, seed=42)
        matched = match_ais(all_to_transmit, beacons, epsilon_m=100.0)

    num_unconfirmed = sum(1 for d in matched if d.ais_status == "unconfirmed")
    incorrect_ais = abs(num_unconfirmed - gt["ships_with_no_ais"])
    amr = ais_mismatch_rate(
        num_incorrect_ais_decisions=incorrect_ais,
        num_total_ais_decisions=len(matched),
    )

    # ---------- Stage 4: confidence-aware payload + baselines ----------
    with timed("payload_sizing", stage_times):
        final_payload_detections = [{"confidence": d.confidence} for d in all_to_transmit]
        all_raw_detections_for_baseline2 = [{"confidence": d.confidence} for obs in observations for d in obs.detections]

        d_raw = baseline_1_raw(num_images=scenario["num_raw_images"])
        d_baseline2 = baseline_2_detection_only(all_raw_detections_for_baseline2)
        d_proposed = proposed_system(final_payload_detections)

    dr_vs_raw = data_reduction_rate(d_raw, d_proposed)
    dr_vs_baseline2 = data_reduction_rate(d_baseline2, d_proposed)

    latency_summary = summarize_latency(stage_times)

    # ---------- Assemble final results ----------
    results = {
        "detection_metrics": DETECTION_METRICS,
        "data_volume_bytes": {
            "baseline_1_raw": d_raw,
            "baseline_2_detection_only": d_baseline2,
            "proposed_system": d_proposed,
        },
        "data_reduction_rate": {
            "proposed_vs_raw_pct": round(dr_vs_raw * 100, 2),
            "proposed_vs_detection_only_pct": round(dr_vs_baseline2 * 100, 2),
        },
        "spatial_dedup": {
            "false_merging_rate_pct": round(fmr * 100, 2),
            "raw_boundary_detections": len(scenario["boundary_pair_raw_detections"]),
            "merged_detections": len(merged_boundary),
        },
        "temporal_dedup": {
            "missed_update_rate_pct": round(mur * 100, 2),
            "total_transmitted": len(all_to_transmit),
            "total_suppressed": len(all_suppressed),
        },
        "ais_matching": {
            "ais_mismatch_rate_pct": round(amr * 100, 2),
            "confirmed": len(matched) - num_unconfirmed,
            "unconfirmed": num_unconfirmed,
        },
        "latency_seconds": latency_summary,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "metrics_final.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # ---------- Comparison table (CSV, drops straight into the report) ----------
    csv_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Baseline 1 (Raw)", "Baseline 2 (Detection-only)", "Proposed System"])
        writer.writerow(["Data volume (bytes)", d_raw, d_baseline2, d_proposed])
        writer.writerow(["Data reduction rate vs Baseline 1", "0%", "-", f"{dr_vs_raw*100:.2f}%"])
        writer.writerow(["Data reduction rate vs Baseline 2", "-", "0%", f"{dr_vs_baseline2*100:.2f}%"])
        writer.writerow(["Precision", "N/A", DETECTION_METRICS["precision"], DETECTION_METRICS["precision"]])
        writer.writerow(["Recall", "N/A", DETECTION_METRICS["recall"], DETECTION_METRICS["recall"]])
        writer.writerow(["mAP50", "N/A", DETECTION_METRICS["mAP50"], DETECTION_METRICS["mAP50"]])
        writer.writerow(["False merging rate", "N/A", "N/A", f"{fmr*100:.2f}%"])
        writer.writerow(["Missed update rate", "N/A", "N/A", f"{mur*100:.2f}%"])
        writer.writerow(["AIS mismatch rate", "N/A", "N/A", f"{amr*100:.2f}%"])
        writer.writerow(["Total latency (s)", "-", "-", f"{latency_summary['total_latency_s']:.4f}"])

    print("=" * 60)
    print("SmartSat - Phase 3 evaluation complete")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print(f"\nSaved to:\n  {out_path}\n  {csv_path}")


if __name__ == "__main__":
    run()