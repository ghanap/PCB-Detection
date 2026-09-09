# WACV PCB Component Detection & SAM Pretrained Point Extractor

State-of-the-Art **Instance Segmentation & Polygon Point Extraction Pipeline** for Printed Circuit Boards (PCBs) using Meta's **Segment Anything Model (SAM)** on the **WACV 2019 PCB Dataset**.

---

## 📌 Features

- **Pretrained SAM Model**: Zero-shot domain transfer using `sam_vit_b.pth`.
- **Bounding Box Prompts**: Automatically prompts SAM using WACV component bounding boxes.
- **Polygon Point Extractor**: Converts SAM binary masks (0s and 1s) into normalized `(x, y)` boundary points using OpenCV contours (`cv2.findContours`).
- **YOLO-seg Dataset Exporter**: Saves annotations in `.txt` format formatted for direct YOLOv8-seg / YOLOv11-seg model training.

---

## 📁 Repository Structure

```text
PCB-Detection/
├── wacv_sam_pipeline.py     # Standalone Python script for SAM polygon point extraction
├── wacv_sam_pipeline.ipynb  # Interactive Jupyter Notebook with visual plots
└── README.md                # Project documentation
```

---

## 🚀 Quick Start

### 1. Run via Python Script
```bash
python wacv_sam_pipeline.py
```

### 2. Run via Jupyter Notebook
Open `wacv_sam_pipeline.ipynb` in VS Code or Jupyter Notebook and run the cells step-by-step.
