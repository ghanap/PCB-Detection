# PCB Component Detection, Segmentation & SAM-Road Graph Studio

### 1. Data Download & Board-Level Split
```bash
python scripts/download_fics_pcb.py
```

### 2. SAM Auto-Annotation (Box-Prompt to YOLO-seg Polygon Masks)
```bash
python scripts/auto_annotate_sam.py
```

### 3. Patch Tiling (640x640 Overlapping Tiles)
```bash
python scripts/tile_dataset.py
```

### 4. Train YOLOv11-seg
```bash
python scripts/train.py
```

### 5. SAM-Road (2024) End-to-End Graph Extraction
Runs SAM ViT Transformer backbone (`sam_vit_b.pth`) for zero-shot domain graph topology extraction.
```bash
python scripts/run_sam_road.py
```
- **Kaggle SAM-Road Notebook**: [notebooks/kaggle_sam_road_pipeline.ipynb](notebooks/kaggle_sam_road_pipeline.ipynb)

## Contributors & Advisors
- **Anagha Pillalamarri** ([@ghanap](https://github.com/ghanap))
- **Prof. Sk Aziz Ali** ([@saali14](https://github.com/saali14)) — Assistant Professor of CSIS, BITS Pilani (Hyderabad)
