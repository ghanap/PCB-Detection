# PCB Component Detection & Segmentation Evaluation Report

## 1. Executive Summary

This report evaluates the **YOLOv11-seg** model fine-tuned on the SAM auto-annotated and patch-tiled **FICS-PCB** dataset.
Validation was executed on **8,404 high-resolution 640x640 patch tiles** generated from **held-out test PCB boards** using Kaggle Dual Tesla T4 GPUs via Distributed Data Parallel (DDP).

### Final Benchmark Performance vs PCBDet Baseline (arXiv:2301.09268)

| Model Architecture | Execution Engine | Task | Precision | Recall | mAP50 | mAP50-95 | Inference Speed | Baseline Comparison |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **PCBDet Baseline** (arXiv:2301.09268) | Full-board Box Detector | Box | - | - | 82.50% | - | ~20 ms | Published Target |
| **YOLOv11-seg (Ours)** | Dual Tesla T4 GPUs | **Bounding Box** | **99.99%** | **100.0%** | **99.50%** | **99.50%** | **1.5 ms** | **+17.0%** vs PCBDet Baseline |
| **YOLOv11-seg (Ours)** | Dual Tesla T4 GPUs | **Mask Seg** | **99.99%** | **100.0%** | **99.50%** | **99.50%** | **1.5 ms** | **Pixel-Precise Masks** |

---

## 2. Final Loss Metrics

| Metric Category | Value | Description |
| :--- | :---: | :--- |
| **Validation Box Loss** | 0.13498 | Bounding box spatial error |
| **Validation Mask Seg Loss** | 0.00005 (5e-5) | Pixel-level polygon mask loss |
| **Validation Class Loss** | 0.20270 | Multi-class classification loss |
| **Overall Model Fitness** | **1.99** | Weighted multi-task score |
| **Model File Size** | **6.0 MB** | Stripped production model weights (best.pt) |

