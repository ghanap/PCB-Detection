# PCB Component Detection & Segmentation

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

## Contributors & Advisors
- **Anagha Pillalamarri** ([@ghanap](https://github.com/ghanap))
- **Prof. Sk Aziz Ali** ([@saali14](https://github.com/saali14)) — Assistant Professor of CSIS, BITS Pilani (Hyderabad)
