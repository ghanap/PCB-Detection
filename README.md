# PCB Component Detection & SAM Point Extractor

SAM-based auto-annotation & polygon point extraction pipeline for PCB component detection.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ghanap/PCB-Detection/blob/main/dataproc.ipynb)

---

## Quick Start

### 1. Open in Colab
[Open dataproc.ipynb in Google Colab](https://colab.research.google.com/github/ghanap/PCB-Detection/blob/main/dataproc.ipynb)

### 2. Import in Kaggle
1. Go to Kaggle Notebooks -> New Notebook -> File -> Import Notebook.
2. Paste: `https://github.com/ghanap/PCB-Detection/blob/main/dataproc.ipynb`

### 3. Run Locally
```bash
python dataproc.py
```

---

## Features

- **SAM Point Extractor**: Extracts polygon boundary points `(x, y)` using SAM.
- **Label Homogenizer**: Maps raw labels to KiCad standard classes.
- **Multi-Format Export**: Saves to YOLO-seg `.txt`, LabelMe `.json`, and `polygon_points.csv`.
- **Data Augmentation**: Spatial flips and HSV color jitter.

---

## Repository Structure

```text
PCB-Detection/
├── dataproc.py        # Python processing script
├── dataproc.ipynb     # Jupyter Notebook
└── README.md          # Documentation
```
