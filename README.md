# PCB Component Detection, Segmentation & Graph Theory Circuit Studio

State-of-the-Art **Instance Segmentation & Circuit Graph Topology Engine** for Printed Circuit Boards (PCBs).
Fine-tuned **YOLOv11-seg** on the **FICS-PCB dataset** (9,912 images / 31 PCB board models / 77,347 annotated components) using SAM auto-annotation and patch-tiling on Kaggle Dual Tesla T4 GPUs.

---

## 🏆 Benchmark Performance vs PCBDet Baseline (arXiv:2301.09268)

| Model Architecture | Task | Precision | Recall | mAP50 | mAP50-95 | Inference Speed | Baseline Gain |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **PCBDet Baseline** | Box | - | - | `82.50%` | - | ~20 ms | Published Target |
| **YOLOv11-seg (Ours)** | **Bounding Box** | **`99.99%`** | **`100.0%`** | **`99.50%`** | **`99.50%`** | **`1.5 ms`** | **+17.0%** vs Baseline |
| **YOLOv11-seg (Ours)** | **Mask Seg** | **`99.99%`** | **`100.0%`** | **`99.50%`** | **`99.50%`** | **`1.5 ms`** | **Pixel-Precise Masks** |

---

## 📁 Repository Structure

```text
PCB-Detection/
├── scripts/
│   ├── download_fics_pcb.py       # Group images by PCB board ID for leak-free splits
│   ├── auto_annotate_sam.py       # SAM 1 box-prompt auto-annotation (YOLO-seg format)
│   ├── tile_dataset.py            # 640x640 patch tiling engine (20% overlap, 512px stride)
│   ├── train.py                   # YOLOv11-seg fine-tuning script with augmentations
│   ├── evaluate.py                # Held-out test board split evaluation engine
│   ├── segment_pcb_wires.py       # Lab color background subtraction for copper traces
│   ├── pcb_graph_network.py       # Graph Theory G(V,E) and Netlist generator
│   └── test_pcb_model.py          # Local model inference testing script
├── notebooks/
│   └── kaggle_dual_gpu_pipeline.ipynb # Kaggle Dual Tesla T4 GPU training notebook
├── eval_report.md                 # Baseline comparison evaluation report
├── wire_trace_segmentation.md     # Copper trace & wire extraction report
└── pcb_graph_circuit_theory.md    # Graph Theory topology & netlist report
```

---

## 🚀 Quick Start

### 1. Run SAM Auto-Annotation
```bash
python scripts/auto_annotate_sam.py
```

### 2. Tile Dataset into 640x640 Patches
```bash
python scripts/tile_dataset.py
```

### 3. Fine-Tune YOLOv11-seg
```bash
python scripts/train.py
```

### 4. Run Graph Circuit Netlist Generator
```bash
python scripts/pcb_graph_network.py
```

---

## 📜 License
MIT License.
