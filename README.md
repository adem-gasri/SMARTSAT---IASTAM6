#  SmartSat — Signal Faible

### IASTAM 6.0 Technical Challenge

**Track 4 — Networks and Communications**
**Problem 7 — Transmitting Information Rather Than Raw Data**

> Intelligent onboard information extraction and relevance-driven transmission system for satellite ship monitoring.

SmartSat aims to reduce satellite downlink volume by processing observations onboard and transmitting **relevant information instead of raw data**.

The approach combines **ship detection, confidence-aware transmission, and spatial, temporal, and AIS-based redundancy reduction**.

---

##  Objective

The project aims to:

* Detect ships directly from satellite imagery.
* Evaluate the relevance and confidence of detections.
* Reduce redundant observations.
* Transmit compact and meaningful information instead of unnecessary raw data.

---

##  System Pipeline

```text
Satellite Image
      │
      ▼
 YOLOv8n Detection
      │
      ▼
Confidence Analysis
      │
      ▼
Redundancy Reduction
      │
      ├── Spatial
      ├── Temporal
      └── AIS-based
      │
      ▼
Relevant Information
      │
      ▼
     Downlink
```

---

##  Ship Detection

The initial detection stage uses **YOLOv8n** trained on a **3,000-image subset of the Airbus Ship Detection Dataset**.

| Parameter  | Value     |
| ---------- | --------- |
| Model      | YOLOv8n   |
| Images     | 3,000     |
| Epochs     | 30        |
| Resolution | 640 × 640 |
| GPU        | NVIDIA T4 |

### Preliminary Results

| Metric    |     Score |
| --------- | --------: |
| Precision | **0.733** |
| Recall    | **0.606** |
| mAP@50    | **0.686** |
| mAP@50–95 | **0.415** |

These results provide the initial baseline for the detection component.

---

##  Project Structure

```text
SmartSat/
│
├── detection/
│   ├── notebooks/
│   │   └── ship_detection_training.ipynb
│   ├── models/
│   │   └── best.pt
│   └── results/
│       ├── detections_example.jpg
│
├── simulation/
│   └── smartsat_simulation.html
│
└── README.md
```

---

##  Status

**Phase 2 — Research and Initial Development**

Current work focuses on the detection baseline, simulation, and initial research methodology.

### Next Steps

* Confidence-aware transmission
* Spatial and temporal redundancy reduction
* AIS-based information filtering
* End-to-end simulation and evaluation

---

## Team

* **Youssef Ben Abdallah**
* **Mohamed Adem Gasri**
* **Faiza Karmous**
* **Emna Filali**

---
