# SmartSat — Signal Faible

**Onboard AI that decides what a satellite sends to the ground, and with how much detail.**

IASTAM 6.0 Technical Challenge · Track 4 — Networks and Communications · Problem 7 — *Transmitting Information Rather Than Raw Data*

---

## Overview

A satellite downlink cannot keep up with the volume of images it captures: contact windows with ground stations are short, and onboard storage and energy are limited. Yet most image tiles contain no ship at all, so sending raw images wastes the scarcest resource of the system.

SmartSat processes each observation **onboard** and transmits only the information that is **new or necessary**. The more uncertain a detection is, the richer the evidence attached to it.

> **Research question** — How can a satellite send only the information that is new or necessary, without losing what maritime surveillance requires?

### Key ideas

| Idea | What it does |
| ---- | ------------ |
| **Overlapping tiling** | 640 × 640 / 768 × 768 tiles with 20 % overlap, so ships on a tile border are not cut. |
| **Onboard detection** | YOLOv8n, light enough for embedded hardware, trained on Airbus ship imagery. |
| **Confidence-aware payload** | High confidence → metadata only; low confidence → image evidence. |
| **Redundancy reduction** | Spatial (NMS + geographic matching), temporal and AIS-based filtering. |

---

## Pipeline

Stages 1–6 run **onboard**; only stage 7 uses the downlink.

```text
 Satellite image
       │
       ▼
 1. Tiling              overlapping tiles (default 640 × 640, 20 % overlap)
       │
       ▼
 2. Detection           YOLOv8n on each tile → global image coordinates
       │
       ▼
 3. Spatial dedup       NMS (IoU 0.4); geographic matching across images (d < Ts)
       │
       ▼
 4. AIS matching        beacon within ε  →  confirmed, otherwise flagged
       │
       ▼
 5. Temporal check      d < Tm  →  ship unchanged, nothing retransmitted
       │
       ▼
 6. Decision            payload size chosen from detection confidence
       │
       ▼
 7. Downlink            reduced payload
```

### Confidence-aware payload

| Tier | Confidence | Transmitted | Size per detection |
| ---- | :--------: | ----------- | -----------------: |
| `high_confidence` | ≥ 0.75 | Metadata only (lat, lon, type, time, confidence) | 64 B |
| `uncertain` | 0.45 – 0.75 | Metadata + small crop | 64 B + 8 KiB |
| `highly_uncertain` | < 0.45 | Metadata + high-resolution crop | 64 B + 60 KiB |

> Sizes are **design values** (`evaluation/baselines.py`), not traffic measured on a real link. Thresholds are configurable and still to be tuned.

> **AIS caveat** — The absence of an AIS beacon is **not** proof of illegitimacy: AIS can be switched off or spoofed. An unmatched ship is *flagged*, never accused.

---

## Results

### Detector (YOLOv8n, validation set)

Source: [`results/metrics_final.json`](results/metrics_final.json), curves in [`detection/results/results_v2.png`](detection/results/results_v2.png).

| Precision | Recall | mAP@50 | mAP@50–95 |
| :-------: | :----: | :----: | :-------: |
| **0.818** | **0.701** | **0.782** | **0.510** |

About 30 % of ships are still missed (recall 0.701) — this is the main limitation of the current detector.

### End-to-end on real images

35 real Airbus images (25 containing ships, 53 ships in total), 20 % tile overlap.
Sources: [`results/real_images_results_tile640.json`](results/real_images_results_tile640.json), [`results/real_images_results_tile768.json`](results/real_images_results_tile768.json).

| | Tile 640 | Tile 768 |
| --- | :---: | :---: |
| True / false positives | 32 / 18 | 36 / 7 |
| Precision | 0.640 | 0.837 |
| Recall | 0.604 | 0.679 |
| Payload sent (crops measured) | 55,950 B | 27,631 B |
| Reduction vs raw (uncompressed) | 99.91 % | 99.96 % |
| Reduction vs JPEG files | 99.07 % | 99.54 % |
| Mean detection time / image | 0.216 s | 0.114 s |

Raw baseline: 61,931,520 B uncompressed (6,047,704 B as JPEG). On this set, 768 × 768 tiles give better precision, recall and payload than 640 × 640, which is why the demo defaults to 768.

### Synthetic multi-pass scenario

45 images, 3 successive observations, simulated AIS ([`results/metrics_final.json`](results/metrics_final.json), [`results/comparison_table.csv`](results/comparison_table.csv)).

| Metric | Baseline 1 (raw images) | Baseline 2 (detections only) | SmartSat |
| ------ | ---: | ---: | ---: |
| Data volume | 79,626,240 B | 448 B | 33,088 B |
| Reduction vs Baseline 1 | — | — | **99.96 %** |
| Reduction vs Baseline 2 | — | — | −7,285 % (sends more) |
| Spatial merges (boundary pair 3 → 2) | — | — | 0 false merges |
| Temporal | — | — | 5 sent, 2 suppressed, 0 missed updates |
| AIS | — | — | 4 confirmed, 1 flagged |
| Post-detection latency | — | — | 0.11 ms |

### How to read these numbers

* The **detections-only** baseline always sends *less* than SmartSat. The extra bytes are the price of **verifiability**: doubtful detections come with image evidence so they can be reviewed on the ground.
* Payload sizes use fixed per-tier values; the crop part of the real-image payload is measured on actual crops. **None of it is traffic measured over a real satellite link.**
* AIS data is **simulated** (`ais/ais_simulator.py`): no real AIS feed was available.
* Latency figures come from a development machine and are **not** representative of flight hardware.
* A single well-detected ship (metadata only, 64 B) gives the most favourable reduction; the average case is lower.

---

## Demo

An interactive Streamlit app runs the pipeline on an uploaded image and shows detections, payload, reduction rate and per-stage timings.

```bash
pip install -r requirements.txt
streamlit run demo_app.py
```

A static, browser-only simulation of the system (no installation) is available in [`simulation/smartsat_simulation.html`](simulation/smartsat_simulation.html), also reachable through `index.html`.

### Command line

```bash
python pipeline.py --image detection/results/detection_example.jpg --show-tiles
```

Writes `results/pipeline_output.json` (what would be transmitted, sizes, timings) and `results/pipeline_output.jpg` (detections coloured by tier).

---

## Reproducing the evaluation

```bash
# Real images: detection, tiling, payload (YOLO-format labels required)
python evaluation/run_real_images.py \
    --images <images_dir> --labels <labels_dir> \
    --weights detection/models/best_v2.pt \
    --tile-size 768 --overlap 0.20 --tag tile768

# Precision / recall / F1 over confidence thresholds
python evaluation/run_evaluation.py \
    --detections results/real_images_detections_tile768.csv \
    --ground-truth results/real_images_ground_truth_tile768_pipeline.json
```

The evaluation images are not stored in the repository (`detection/test_images/` is git-ignored). Result files are tagged by tile size (`tile640`, `tile768`).

---

## Repository structure

```text
.
├── pipeline.py                  # End-to-end pipeline: tiling → detection → dedup → payload
├── demo_app.py                  # Streamlit demo
├── index.html                   # Entry point to the browser simulation
├── requirements.txt
│
├── detection/
│   ├── detect.py                # ShipDetector: thin YOLOv8n wrapper
│   ├── models/                  # best.pt (first run), best_v2.pt (final)
│   ├── notebooks/               # ship_detection_training.ipynb (Kaggle, T4)
│   └── results/                 # example detections, v2 training curves
│
├── spatial_dedup/
│   ├── tiling.py                # overlapping tiles, tile → global coordinates
│   ├── nms_merge.py             # NMS across tiles (IoU 0.4)
│   └── geo_match.py             # geographic matching across images (Ts)
│
├── temporal/
│   ├── detection_list.py        # Detection / DetectionList data structures
│   └── temporal_compare.py      # unchanged vs displaced vs new (Tm)
│
├── ais/
│   ├── ais_matcher.py           # beacon matching within ε (haversine)
│   └── ais_simulator.py         # synthetic AIS beacons
│
├── evaluation/
│   ├── baselines.py             # Baseline 1, Baseline 2, payload tiers
│   ├── metrics.py               # reduction rate, timing helpers
│   ├── scenarios.py             # reproducible synthetic scenario
│   ├── run_evaluation.py        # precision / recall / F1 sweep
│   └── run_real_images.py       # real-image evaluation
│
├── simulation/
│   ├── satellite_sim.py         # battery, memory, contact windows
│   └── smartsat_simulation.html # browser simulation
│
└── results/                     # JSON / CSV outputs of all experiments
```

---

## Detector training

| Parameter | Value |
| --------- | ----- |
| Model | YOLOv8n (pretrained) |
| Data | 3,000-image subset of the Airbus Ship Detection dataset, 80 / 20 split |
| Image size | 640 × 640 |
| Hardware | NVIDIA T4 (Kaggle) |
| Notebook | [`detection/notebooks/ship_detection_training.ipynb`](detection/notebooks/ship_detection_training.ipynb) |

---

## Limitations and next steps

**Known limitations**

* Recall of 0.70: about 30 % of ships are missed; precision drops on cluttered scenes.
* Confidence is not certainty — the tier boundaries (0.75 / 0.45) have not been calibrated.
* A geolocation error can create false "movement" in the temporal comparison.
* The temporal and AIS modules are validated on a **synthetic** scenario; they are not yet integrated in `pipeline.py` or the demo.
* AIS reception is assumed reliable, while spoofing is possible.

**Next steps**

* Train on the full dataset and improve recall.
* Study the effect of tile size and overlap on recall.
* Calibrate confidence thresholds, `Ts`, `Tm` and `ε` experimentally.
* Integrate temporal comparison and AIS matching into the main pipeline.
* Measure latency and computation on embedded-class hardware.

---

## Team

**Team Signal Faible — FST**

* Youssef Ben Abdallah
* Mohamed Adem Gasri
* Faiza Karmous
* Emna Filali
* Mohamed Amine Louati

## References

1. ESA — onboard processing on COTS hardware.
2. arXiv:2306.00275.
3. Redmon et al., *You Only Look Once: Unified, Real-Time Object Detection*, CVPR 2016.
4. Zhao et al., 2024.
5. Li et al., 2022.
6. arXiv:2409.14726 — semantic communication for 6G non-terrestrial networks.
7. Yang et al., 2024.
8. IEEE, 2024.
