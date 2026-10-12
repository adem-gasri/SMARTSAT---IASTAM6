
import argparse
import csv
import json
from pathlib import Path

from spatial_dedup.nms_merge import iou


def calculate_metrics(detections, ground_truth, confidence_threshold, iou_threshold=0.5):
    # Garder uniquement les détections au-dessus du seuil
    detections = [
        d for d in detections
        if float(d["confidence"]) >= confidence_threshold
    ]
    detections.sort(key=lambda d: float(d["confidence"]), reverse=True)

    used_gt = set()
    tp = 0

    # Association des détections aux vrais navires
    for detection in detections:
        best_index = None
        best_iou = iou_threshold

        for index, gt_box in enumerate(ground_truth):
            if index in used_gt:
                continue

            score = iou(detection["bbox_xyxy"], gt_box)

            if score >= best_iou:
                best_iou = score
                best_index = index

        if best_index is not None:
            tp += 1
            used_gt.add(best_index)

    fp = len(detections) - tp
    fn = len(ground_truth) - tp

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "threshold": confidence_threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "F1": round(f1, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument(
        "--thresholds",
        default="0.10,0.20,0.25,0.30,0.40,0.50,0.60,0.75"
    )
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    detections_by_image = {}

    with open(args.detections, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required = {"image", "confidence", "x1", "y1", "x2", "y2"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Le CSV doit contenir image, confidence, x1, y1, x2, y2. "
                "Regénère-le avec run_real_images.py."
            )

        for row in reader:
            detection = {
                "confidence": float(row["confidence"]),
                "bbox_xyxy": [
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                ],
            }
            detections_by_image.setdefault(row["image"], []).append(detection)

    with open(args.ground_truth, encoding="utf-8") as file:
        gt_data = json.load(file)

    gt_by_image = gt_data["ground_truth"]
    iou_threshold = float(gt_data.get("iou_threshold", 0.5))
    thresholds = [
        float(value.strip())
        for value in args.thresholds.split(",")
        if value.strip()
    ]

    if not thresholds or any(t < 0 or t > 1 for t in thresholds):
        raise ValueError("Les seuils doivent être compris entre 0 et 1.")

    images = sorted(set(gt_by_image) | set(detections_by_image))
    results = []

    for threshold in thresholds:
        total = {
            "TP": 0,
            "FP": 0,
            "FN": 0,
        }

        for image in images:
            metrics = calculate_metrics(
                detections_by_image.get(image, []),
                gt_by_image.get(image, []),
                threshold,
                iou_threshold,
            )

            for key in total:
                total[key] += metrics[key]

        tp, fp, fn = total["TP"], total["FP"], total["FN"]
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        results.append({
            "confidence_threshold": threshold,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "F1": round(f1, 4),
            "images_evaluated": len(images),
        })

    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = Path(args.detections).stem.replace(
        "real_images_detections_", ""
    )

    csv_path = output_dir / f"threshold_sweep_{tag}.csv"
    json_path = output_dir / f"threshold_sweep_{tag}.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump({
            "yolo_rerun": False,
            "iou_threshold": iou_threshold,
            "results": results,
        }, file, indent=2)

    print("\nComparaison des seuils")
    for result in results:
        print(
            f"Seuil={result['confidence_threshold']:.2f} | "
            f"Precision={result['precision']:.3f} | "
            f"Recall={result['recall']:.3f} | "
            f"F1={result['F1']:.3f} | "
            f"TP={result['TP']} FP={result['FP']} FN={result['FN']}"
        )

    print(f"\nCSV sauvegardé : {csv_path}")
    print(f"JSON sauvegardé : {json_path}")


if __name__ == "__main__":
    main()
