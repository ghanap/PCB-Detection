# WACV PCB Component Detection & SAM Pretrained Point Extractor

State-of-the-Art **Instance Segmentation & Polygon Point Extraction Pipeline** for Printed Circuit Boards (PCBs) using Meta's **Segment Anything Model (SAM)** on the **WACV 2019 PCB Dataset**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ghanap/PCB-Detection/blob/main/wacv_sam_pipeline.ipynb)

---

## 📌 How to Open & Run in Cloud GPUs

### Option 1: Open in Google Colab (1-Click)
Click the badge above or use this link:  
👉 **[Open wacv_sam_pipeline.ipynb in Google Colab](https://colab.research.google.com/github/ghanap/PCB-Detection/blob/main/wacv_sam_pipeline.ipynb)**

### Option 2: Open in Kaggle Notebooks (Dual Tesla T4 GPUs)
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code).
2. Click **New Notebook** -> **File** -> **Import Notebook**.
3. Paste the GitHub notebook URL:
   ```text
   https://github.com/ghanap/PCB-Detection/blob/main/wacv_sam_pipeline.ipynb
   ```
4. Click **Import** and turn on **Accelerator: GPU (T4 x 2)** in the right sidebar!

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

## 🚀 Quick Start (Local)

```bash
python wacv_sam_pipeline.py
```
