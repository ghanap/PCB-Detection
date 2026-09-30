#!/usr/bin/env python3
"""
clean-and-segment-pipeline.py (No underscores)

Step 1: Data Exploration (stats, class counts, box diagnostics)
Step 2: Data Cleaning (clamp bounds, fix inversions, filter noise/text, map to KiCad KLC)
Step 3: Run SAM on top of cleaned boxes (OBB & BB polygon extraction)
Step 4: Save cleaned datasets, CSV/Excel tables, and LabelMe JSONs
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Any

import cv2
import numpy as np
import pandas as pd
import torch

# KiCad Footprint Mapping
KAGGLE_CLASSES = {
    0: {'raw': 'Cap1',        'kicad': 'Capacitor-SMD:C-0805-2012Metric', 'prefix': 'C'},
    1: {'raw': 'Cap2',        'kicad': 'Capacitor-SMD:C-0805-2012Metric', 'prefix': 'C'},
    2: {'raw': 'Cap3',        'kicad': 'Capacitor-SMD:C-0805-2012Metric', 'prefix': 'C'},
    3: {'raw': 'Cap4',        'kicad': 'Capacitor-SMD:C-0805-2012Metric', 'prefix': 'C'},
    4: {'raw': 'MOSFET',      'kicad': 'Package-TO-SOT-SMD:SOT-23',      'prefix': 'Q'},
    5: {'raw': 'Mov',         'kicad': 'Diode-SMD:D-SOD-123',             'prefix': 'D'},
    6: {'raw': 'Resistor',    'kicad': 'Resistor-SMD:R-0805-2012Metric', 'prefix': 'R'},
    7: {'raw': 'Transformer', 'kicad': 'Transformer-SMD:WE-PoE-EP13',     'prefix': 'T'}
}

WACV_NOISE_TAGS = {'text', 'pads', 'pins', 'test', 'unknown', 'jumper', 'fiducial'}

WACV_CLASSES = {
    'resistor':     {'kicad': 'Resistor-SMD:R-0805-2012Metric', 'prefix': 'R'},
    'capacitor':    {'kicad': 'Capacitor-SMD:C-0805-2012Metric', 'prefix': 'C'},
    'electrolytic': {'kicad': 'Capacitor-THT:CP-Radial-D8mm',   'prefix': 'C'},
    'ic':           {'kicad': 'Package-SO:SOIC-8-3.9x4.9mm',    'prefix': 'U'},
    'component':    {'kicad': 'Package-SO:SOIC-8-3.9x4.9mm',    'prefix': 'U'},
    'transistor':   {'kicad': 'Package-TO-SOT-SMD:SOT-23',      'prefix': 'Q'},
    'diode':        {'kicad': 'Diode-SMD:D-SOD-123',             'prefix': 'D'},
    'inductor':     {'kicad': 'Inductor-SMD:L-0805-2012Metric', 'prefix': 'L'},
    'connector':    {'kicad': 'Connector-PinHeader:1x04',        'prefix': 'J'},
    'led':          {'kicad': 'LED-SMD:LED-0805-2012Metric',    'prefix': 'D'},
    'switch':       {'kicad': 'Button-Switch-SMD:SW-SPST',      'prefix': 'SW'},
    'button':       {'kicad': 'Button-Switch-SMD:SW-Push',      'prefix': 'SW'}
}

def clean_box(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int, min_dim: int = 6) -> Tuple[bool, List[int]]:
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    x1 = max(0, min(int(round(x1)), img_w - 1))
    y1 = max(0, min(int(round(y1)), img_h - 1))
    x2 = max(0, min(int(round(x2)), img_w))
    y2 = max(0, min(int(round(y2)), img_h))
    if (x2 - x1) < min_dim or (y2 - y1) < min_dim:
        return False, []
    return True, [x1, y1, x2, y2]

def extract_polygon(mask: np.ndarray, min_area: float = 12.0) -> List[List[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return []
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area: return []
    approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
    return [[round(float(p[0][0]), 2), round(float(p[0][1]), 2)] for p in approx]

def explore_and_clean_kaggle(data_dir: Path, out_dir: Path, max_images: int = 50):
    print("\n" + "=" * 75)
    print("STEP 1: DATA EXPLORATION - KAGGLE / FICS DATASET")
    print("=" * 75)

    img_dir = data_dir / "test" / "images"
    lbl_dir = data_dir / "test" / "labels"
    if not img_dir.exists():
        img_dir = data_dir / "train" / "images"
        lbl_dir = data_dir / "train" / "labels"

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    print(f"Total available images in split: {len(img_files)}")

    class_counts = {}
    box_issues = {"inverted": 0, "out_of_bounds": 0, "too_small": 0, "valid": 0}

    for p in img_files[:200]:
        lbl_p = lbl_dir / f"{p.stem}.txt"
        if not lbl_p.exists(): continue
        with open(lbl_p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(float(parts[0]))
                    class_counts[cid] = class_counts.get(cid, 0) + 1
                    xc, yc, bw, bh = map(float, parts[1:5])
                    if bw <= 0 or bh <= 0: box_issues["too_small"] += 1
                    elif xc < 0 or xc > 1 or yc < 0 or yc > 1: box_issues["out_of_bounds"] += 1
                    else: box_issues["valid"] += 1

    print("\nClass Distribution (Raw Dataset):")
    for cid, cnt in sorted(class_counts.items()):
        name = KAGGLE_CLASSES.get(cid, {}).get('raw', f'Class-{cid}')
        print(f"  [{cid}] {name:15s}: {cnt:5d} instances")

    print("\nData Quality Diagnostics:")
    print(f"  Valid Bounding Boxes:     {box_issues['valid']}")
    print(f"  Corrupt / Out of Bounds:  {box_issues['out_of_bounds']}")
    print(f"  Degenerate / Zero Area:   {box_issues['too_small']}")
    print("-" * 75)

    # Clean and save
    out_clean_images = out_dir / "kaggle-cleaned" / "images"
    out_clean_labels = out_dir / "kaggle-cleaned" / "labels"
    out_clean_images.mkdir(parents=True, exist_ok=True)
    out_clean_labels.mkdir(parents=True, exist_ok=True)

    cleaned_samples = []
    for img_p in img_files[:max_images]:
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if not lbl_p.exists(): continue
        img = cv2.imread(str(img_p))
        if img is None: continue
        h, w = img.shape[:2]

        clean_lines = []
        clean_boxes = []
        with open(lbl_p, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(float(parts[0]))
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = (xc - bw / 2.0) * w
                    y1 = (yc - bh / 2.0) * h
                    x2 = (xc + bw / 2.0) * w
                    y2 = (yc + bh / 2.0) * h
                    ok, box = clean_box(x1, y1, x2, y2, w, h)
                    if ok:
                        clean_xc = ((box[0] + box[2]) / 2.0) / w
                        clean_yc = ((box[1] + box[3]) / 2.0) / h
                        clean_bw = (box[2] - box[0]) / w
                        clean_bh = (box[3] - box[1]) / h
                        clean_lines.append(f"{cid} {clean_xc:.6f} {clean_yc:.6f} {clean_bw:.6f} {clean_bh:.6f}")
                        clean_boxes.append((box, cid))

        if clean_lines:
            shutil.copy2(img_p, out_clean_images / img_p.name)
            with open(out_clean_labels / f"{img_p.stem}.txt", "w") as f:
                f.write("\n".join(clean_lines))
            cleaned_samples.append((img_p, clean_boxes, (h, w)))

    print(f"SUCCESS: Saved {len(cleaned_samples)} cleaned images & labels into {out_dir / 'kaggle-cleaned'}")
    return cleaned_samples

def explore_and_clean_wacv(wacv_dir: Path, out_dir: Path, max_boards: int = 5):
    import xml.etree.ElementTree as ET
    print("\n" + "=" * 75)
    print("STEP 2: DATA EXPLORATION - WACV 2019 INDUSTRIAL PCB DATASET")
    print("=" * 75)

    board_dirs = sorted([d for d in wacv_dir.iterdir() if d.is_dir() and list(d.glob("*.xml"))])
    print(f"Total PCB boards found: {len(board_dirs)}")

    wacv_tag_counts = {}
    skipped_text = 0
    valid_boxes = 0

    for b in board_dirs:
        xml_file = list(b.glob("*.xml"))[0]
        root = ET.parse(xml_file).getroot()
        for obj in root.findall("object"):
            raw_tag = obj.find("name").text.strip().lower() if obj.find("name") is not None else ""
            first_w = raw_tag.split()[0].strip('"') if raw_tag else ""
            wacv_tag_counts[first_w] = wacv_tag_counts.get(first_w, 0) + 1
            if first_w in WACV_NOISE_TAGS:
                skipped_text += 1
            else:
                valid_boxes += 1

    print("\nTop Component & Label Tags in WACV Annotations:")
    for tag, c in sorted(wacv_tag_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
        status = "[NOISE/FILTERED]" if tag in WACV_NOISE_TAGS else "[COMPONENT]"
        print(f"  {tag:15s}: {c:5d}  {status}")

    print(f"\nFiltering Summary:")
    print(f"  Noise / Silkscreen tags filtered out: {skipped_text:,}")
    print(f"  Valid Electronic Components retained:  {valid_boxes:,}")

    # Clean boards
    out_wacv_clean = out_dir / "wacv-cleaned"
    out_wacv_clean.mkdir(parents=True, exist_ok=True)
    cleaned_boards = []

    for b in board_dirs[:max_boards]:
        xml_file = list(b.glob("*.xml"))[0]
        img_matches = list(b.glob("*.jpg")) + list(b.glob("*.png"))
        if not img_matches: continue
        img_p = img_matches[0]
        img = cv2.imread(str(img_p))
        if img is None: continue
        h, w = img.shape[:2]

        root = ET.parse(xml_file).getroot()
        board_clean_boxes = []
        for obj in root.findall("object"):
            raw_tag = obj.find("name").text.strip().lower() if obj.find("name") is not None else ""
            first_w = raw_tag.split()[0].strip('"') if raw_tag else ""
            if first_w in WACV_NOISE_TAGS: continue

            bnd = obj.find("bndbox")
            if bnd is None: continue
            x1, y1 = float(bnd.find("xmin").text), float(bnd.find("ymin").text)
            x2, y2 = float(bnd.find("xmax").text), float(bnd.find("ymax").text)
            ok, box = clean_box(x1, y1, x2, y2, w, h)
            if ok:
                info = WACV_CLASSES.get(first_w, WACV_CLASSES['component'])
                board_clean_boxes.append((box, first_w, info))

        cleaned_boards.append((b.name, img_p, board_clean_boxes, (h, w)))

    print(f"SUCCESS: Cleaned and filtered {len(cleaned_boards)} boards for SAM segmentation.")
    return cleaned_boards

def run_sam_on_cleaned(kaggle_samples, wacv_samples, sam_path: Path, out_dir: Path):
    from segment_anything import sam_model_registry, SamPredictor

    print("\n" + "=" * 75)
    print("STEP 3: RUN SAM ON TOP OF CLEANED BOUNDING BOXES")
    print("=" * 75)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM ({sam_path}) on {device}...")
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_path))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    print("SAM ViT-B predictor ready.")

    out_points_dir = out_dir / "sam-segmented-results"
    out_points_dir.mkdir(parents=True, exist_ok=True)

    all_records = []

    # 1. Process cleaned Kaggle images
    print(f"\nRunning SAM on {len(kaggle_samples)} Cleaned Kaggle Images...")
    for idx, (img_p, boxes, (h, w)) in enumerate(kaggle_samples, 1):
        img = cv2.imread(str(img_p))
        predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        shapes = []
        ref_counts = {}

        for b_idx, (b, cid) in enumerate(boxes, 1):
            masks, scores, _ = predictor.predict(box=np.array(b)[None, :], multimask_output=False)
            conf = float(scores[0])
            pts = extract_polygon(masks[0])
            if not pts:
                pts = [[float(b[0]), float(b[1])], [float(b[2]), float(b[1])], [float(b[2]), float(b[3])], [float(b[0]), float(b[3])]]

            info = KAGGLE_CLASSES.get(cid, KAGGLE_CLASSES[6])
            pfx = info['prefix']
            ref_counts[pfx] = ref_counts.get(pfx, 0) + 1
            ref_des = f"{pfx}{ref_counts[pfx]}"

            all_records.append({
                "dataset": "Kaggle-FICS",
                "image_or_board": img_p.name,
                "instance_id": b_idx,
                "ref_des": ref_des,
                "class_name": info['raw'],
                "kicad_footprint": info['kicad'],
                "box_x1": b[0], "box_y1": b[1], "box_x2": b[2], "box_y2": b[3],
                "confidence": round(conf, 3),
                "num_polygon_points": len(pts),
                "polygon_points": json.dumps(pts)
            })
            shapes.append({"label": f"{ref_des}: {info['raw']}", "points": pts, "shape_type": "polygon"})

        shutil.copy2(img_p, out_points_dir / img_p.name)
        with open(out_points_dir / f"{img_p.stem}.json", "w", encoding="utf-8") as f:
            json.dump({"version": "5.5.0", "flags": {}, "shapes": shapes, "imagePath": img_p.name, "imageHeight": h, "imageWidth": w}, f, indent=2)

    # 2. Process cleaned WACV boards
    print(f"\nRunning SAM on {len(wacv_samples)} Cleaned WACV Boards...")
    for idx, (board_name, img_p, boxes, (h, w)) in enumerate(wacv_samples, 1):
        img = cv2.imread(str(img_p))
        predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        shapes = []
        ref_counts = {}

        for b_idx, (b, tag, info) in enumerate(boxes, 1):
            masks, scores, _ = predictor.predict(box=np.array(b)[None, :], multimask_output=False)
            conf = float(scores[0])
            pts = extract_polygon(masks[0])
            if not pts:
                pts = [[float(b[0]), float(b[1])], [float(b[2]), float(b[1])], [float(b[2]), float(b[3])], [float(b[0]), float(b[3])]]

            pfx = info['prefix']
            ref_counts[pfx] = ref_counts.get(pfx, 0) + 1
            ref_des = f"{pfx}{ref_counts[pfx]}"

            all_records.append({
                "dataset": "WACV-2019",
                "image_or_board": board_name,
                "instance_id": b_idx,
                "ref_des": ref_des,
                "class_name": tag,
                "kicad_footprint": info['kicad'],
                "box_x1": b[0], "box_y1": b[1], "box_x2": b[2], "box_y2": b[3],
                "confidence": round(conf, 3),
                "num_polygon_points": len(pts),
                "polygon_points": json.dumps(pts)
            })
            shapes.append({"label": f"{ref_des}: {tag.upper()}", "points": pts, "shape_type": "polygon"})

        shutil.copy2(img_p, out_points_dir / img_p.name)
        with open(out_points_dir / f"{img_p.stem}.json", "w", encoding="utf-8") as f:
            json.dump({"version": "5.5.0", "flags": {}, "shapes": shapes, "imagePath": img_p.name, "imageHeight": h, "imageWidth": w}, f, indent=2)
        print(f"  Board [{board_name}] done: {len(boxes)} components segmented.")

    # Save summary tables
    df = pd.DataFrame(all_records)
    csv_out = out_points_dir / "all-cleaned-sam-points.csv"
    xlsx_out = out_points_dir / "all-cleaned-sam-points.xlsx"
    df.to_csv(csv_out, index=False)
    df.to_excel(xlsx_out, index=False)

    print("\n" + "=" * 75)
    print("STEP 4: OUTPUT TABLES & SAVED DELIVERABLES")
    print("=" * 75)
    print(f"Total Components Extracted across both datasets: {len(all_records)}")
    print(f"Saved Cleaned CSV:   {csv_out}")
    print(f"Saved Cleaned Excel: {xlsx_out}")
    print(f"Saved LabelMe JSONs: {out_points_dir}")
    print("=" * 75)

if __name__ == "__main__":
    BASE = Path(r"c:\Userdata\antiiii")
    KAGGLE_PATH = BASE / "dataset_split"
    WACV_PATH = BASE / "wacv_data" / "pcb_wacv_2019"
    SAM_PATH = BASE / "sam_vit_b.pth"
    OUT_PATH = BASE / "cleaned-pipeline-output"

    k_samples = explore_and_clean_kaggle(KAGGLE_PATH, OUT_PATH, max_images=25)
    w_samples = explore_and_clean_wacv(WACV_PATH, OUT_PATH, max_boards=3)
    run_sam_on_cleaned(k_samples, w_samples, SAM_PATH, OUT_PATH)
