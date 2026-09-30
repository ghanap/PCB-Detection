# PCB Component Detection & SAM Polygon Point Extractor

Pipeline for Printed Circuit Board (PCB) component segmentation, polygon point extraction, and KiCad footprint mapping using Meta's Segment Anything Model (SAM ViT-B).

## Repository Structure

```text
PCB-Detection/
├── run_sam.py              # Ingests PCB images & bboxes, extracts SAM polygon points, exports CSV/Excel & LabelMe
├── run_sam.ipynb           # Notebook for SAM point extraction
│
├── clean_kaggle.py         # Cleaning & SAM extraction pipeline for Kaggle / FICS dataset
├── clean_kaggle.ipynb      # Notebook for Kaggle dataset cleaning
│
├── clean_wacv.py           # Cleaning & SAM extraction pipeline for WACV 2019 dataset
├── clean_wacv.ipynb        # Notebook for WACV dataset cleaning
│
├── outputs/                # Generated point spreadsheets
│   ├── kaggle_sam_output_points.csv
│   ├── kaggle_sam_output_points.xlsx
│   ├── wacv_arduino_sam_points.csv
│   └── wacv_arduino_sam_points.xlsx
│
├── README.md               # Documentation
└── .gitignore
```

## Usage

### 1. Run SAM & Extract Polygon Points
```bash
python run_sam.py --image <path_to_image> --labels <path_to_yolo_labels>
```
* Prompts SAM with component bounding boxes.
* Extracts polygon boundary points `[(x, y), ...]`.
* Saves results to `sam_output_points.xlsx` and `sam_output_points.csv`.
* Creates LabelMe JSON with `Ref_Des: Class` tags.

### 2. Clean Kaggle / FICS Dataset
```bash
python clean_kaggle.py
```
* Cleans and validates bounding boxes (`Cap1`–`Cap4`, `MOSFET`, `Mov`, `Resistor`, `Transformer`).
* Maps classes to KiCad Library Convention (KLC) footprints.
* Exports summary spreadsheets and YOLO-seg labels.

### 3. Clean WACV 2019 Dataset
```bash
python clean_wacv.py --board-dir <path_to_wacv_board_folder>
```
* Parses Pascal VOC XML annotations, skipping silkscreen `text` and `pads`.
* Cleans bounding boxes (`ic`, `connector`, `led`, `resistor`, `capacitor`, `diode`, `clock`, etc.).
* Extracts SAM polygon masks and exports to CSV, Excel, and LabelMe.

## Output Format

Output CSV and Excel spreadsheets contain:
* `ref_des`: Reference Designator (`R1`, `C1`, `U1`, `J1`, `Q1`, `T1`...)
* `class_name`: Component class
* `kicad_footprint`: KiCad footprint identifier
* `confidence`: SAM segmentation confidence score
* `num_polygon_points`: Number of vertices
* `polygon_points_compact`: String format `(x1,y1); (x2,y2); ...`
* `polygon_points_json`: JSON coordinate array `[[x1, y1], [x2, y2], ...]`

## Reproducibility & Benchmark Training

The full multi-model benchmark (YOLO bounding boxes vs SAM pseudo-labels) was trained and evaluated on Kaggle GPUs. 

To completely recreate the 10-hour training run and evaluation metrics side-by-side, you can run this Kaggle notebook with a single click:
* **Kaggle Notebook:** [PCB YOLO & SAM Multi-Model Training](https://www.kaggle.com/code/dragonfliez/pcb-yolo-sam-multi-model-training)
* **Dataset Used:** [FICS PCB YOLO Cleaned](https://www.kaggle.com/datasets/dragonfliez/fics-pcb-yolo-cleaned)

*(Note: If the links return a 404 error, the author needs to toggle their Kaggle privacy settings to 'Public'.)*
