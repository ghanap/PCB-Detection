# PCB Component Detection & SAM Point Extractor

SAM-based auto-annotation and polygon point extraction for PCB component detection.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ghanap/PCB-Detection/blob/main/dataproc.ipynb)

## Quick Start

### Google Colab
[Launch in Colab](https://colab.research.google.com/github/ghanap/PCB-Detection/blob/main/dataproc.ipynb)

### Kaggle
Import Notebook URL:
`https://github.com/ghanap/PCB-Detection/blob/main/dataproc.ipynb`

### Local
```bash
python dataproc.py
```

## Features

- **SAM Polygon Point Extractor**: Extracts boundary points `(x, y)` from Segment Anything Model masks.
- **Label Homogenizer**: Maps multi-dataset labels into KiCad Library Convention (KLC) footprint names (`Capacitor_SMD`, `Resistor_SMD`, `Package_SO`, etc.).
- **Multi-Format Export**: Outputs YOLO-seg `.txt`, LabelMe `.json`, and `polygon_points.csv`.
- **Data Augmentation**: Spatial flips and HSV color jitter.

## Directory Structure

```text
PCB-Detection/
├── dataproc.py        # Pipeline script
├── dataproc.ipynb     # Interactive notebook
└── README.md          # Project documentation
```
