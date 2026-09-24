#!/usr/bin/env python3
"""
clean_kaggle.py

Pipeline 1 of 2: Data Cleaning for Kaggle / FICS PCB Dataset
- Classes: Cap1, Cap2, Cap3, Cap4, MOSFET, Mov, Resistor, Transformer
- Cleans bounding boxes (boundary clamping, coordinate inversion fix, degenerate box removal).
- Prompts SAM (Segment Anything Model) with cleaned boxes.
- Extracts polygon boundary points [(x, y), ...] using Douglas-Peucker contour simplification.
- Standardizes labels to KiCad Library Convention (KLC).
- Exports cleaned YOLO-seg labels, images, and summary XLSX / CSV.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any

import cv2
import numpy as np
import pandas as pd
import torch
from segment_anything import sam_model_registry, SamPredictor

KAGGLE_CLASSES = {
    0: {'raw': 'Cap1',        'kicad': 'Capacitor_SMD',      'footprint': 'Capacitor_SMD:C_0805_2012Metric', 'ref_prefix': 'C'},
    1: {'raw': 'Cap2',        'kicad': 'Capacitor_SMD',      'footprint': 'Capacitor_SMD:C_0805_2012Metric', 'ref_prefix': 'C'},
    2: {'raw': 'Cap3',        'kicad': 'Capacitor_SMD',      'footprint': 'Capacitor_SMD:C_0805_2012Metric', 'ref_prefix': 'C'},
    3: {'raw': 'Cap4',        'kicad': 'Capacitor_SMD',      'footprint': 'Capacitor_SMD:C_0805_2012Metric', 'ref_prefix': 'C'},
    4: {'raw': 'MOSFET',      'kicad': 'Package_TO_SOT_SMD', 'footprint': 'Package_TO_SOT_SMD:SOT-23',      'ref_prefix': 'Q'},
    5: {'raw': 'Mov',         'kicad': 'Diode_SMD',          'footprint': 'Diode_SMD:D_SOD-123',             'ref_prefix': 'D'},
    6: {'raw': 'Resistor',    'kicad': 'Resistor_SMD',       'footprint': 'Resistor_SMD:R_0805_2012Metric', 'ref_prefix': 'R'},
    7: {'raw': 'Transformer', 'kicad': 'Transformer_SMD',    'footprint': 'Transformer_SMD:Transformer_Wuerth_WE-PoE_EP13', 'ref_prefix': 'T'}
}

def clean_box(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int, min_size: int = 8) -> Tuple[bool, List[int]]:
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    x1 = max(0, min(int(round(x1)), img_w - 1))
    y1 = max(0, min(int(round(y1)), img_h - 1))
    x2 = max(0, min(int(round(x2)), img_w - 1))
    y2 = max(0, min(int(round(y2)), img_h - 1))
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return False, []
    return True, [x1, y1, x2, y2]

def extract_polygon(mask: np.ndarray, min_area: float = 15.0) -> List[List[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return []
    epsilon = 0.005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    return [[float(p[0][0]), float(p[0][1])] for p in approx]

def clean_kaggle_dataset(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    sam_checkpoint: Path,
    limit: int = None
):
    print("=" * 70)
    print("[1/2] KAGGLE PCB DATA CLEANING PIPELINE")
    print("=" * 70)

    out_img_dir = output_dir / "images"
    out_yolo_dir = output_dir / "labels_yolo_seg"
    out_vis_dir = output_dir / "visual_checks"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_yolo_dir.mkdir(parents=True, exist_ok=True)
    out_vis_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM ({sam_checkpoint}) on {device}...")
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    img_files = sorted([p for p in images_dir.glob("*.jpg") if not any(t in p.name for t in ["_mask", "_overlay", "_sam", "_seg_vis"])])
    if limit:
        img_files = img_files[:limit]

    print(f"Found {len(img_files)} Kaggle images to clean...")
    summary_rows = []
    total_in = 0
    total_out = 0

    for idx, img_p in enumerate(img_files, 1):
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = img_p.stem
        lbl_p = labels_dir / f"{stem}.txt"
        if not lbl_p.exists():
            continue

        raw_boxes = []
        with open(lbl_p, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(float(parts[0]))
                    xc, yc, bw, bh = map(float, parts[1:5])
                    raw_boxes.append(((xc - bw/2)*w, (yc - bh/2)*h, (xc + bw/2)*w, (yc + bh/2)*h, cid))

        if not raw_boxes:
            continue
        total_in += len(raw_boxes)

        valid_boxes = []
        for (x1, y1, x2, y2, cid) in raw_boxes:
            ok, box = clean_box(x1, y1, x2, y2, w, h)
            if ok:
                valid_boxes.append((box, cid))

        if not valid_boxes:
            continue

        predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        vis_img = img.copy()
        yolo_seg_rows = []
        ref_counters = {}

        for inst_id, (box, cid) in enumerate(valid_boxes, 1):
            total_out += 1
            input_box = np.array(box)
            masks, scores, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
            conf = float(scores[0])

            pts = extract_polygon(masks[0])
            if not pts:
                pts = [[float(box[0]), float(box[1])], [float(box[2]), float(box[1])],
                       [float(box[2]), float(box[3])], [float(box[0]), float(box[3])]]

            info = KAGGLE_CLASSES.get(cid, KAGGLE_CLASSES[6])
            pfx = info['ref_prefix']
            ref_counters[pfx] = ref_counters.get(pfx, 0) + 1
            ref_des = f"{pfx}{ref_counters[pfx]}"

            norm_pts_str = " ".join([f"{p[0]/w:.6f} {p[1]/h:.6f}" for p in pts])
            yolo_seg_rows.append(f"{cid} {norm_pts_str}")

            summary_rows.append({
                "image": img_p.name,
                "instance_id": inst_id,
                "ref_des": ref_des,
                "raw_label": info['raw'],
                "kicad_class": info['kicad'],
                "kicad_footprint": info['footprint'],
                "box_x1": box[0], "box_y1": box[1], "box_x2": box[2], "box_y2": box[3],
                "confidence": round(conf, 3),
                "num_polygon_points": len(pts),
                "polygon_points": json.dumps(pts)
            })

            cnt = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_img, [cnt], True, (0, 255, 255), 2)
            cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
            cv2.putText(vis_img, ref_des, (box[0], max(18, box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        shutil.copy2(img_p, out_img_dir / img_p.name)
        with open(out_yolo_dir / f"{stem}.txt", 'w') as f:
            f.write("\n".join(yolo_seg_rows))
        cv2.imwrite(str(out_vis_dir / f"{stem}_cleaned.jpg"), vis_img)
        print(f"[{idx}/{len(img_files)}] Cleaned {img_p.name} -> {len(valid_boxes)} components")

    # Export to XLSX & CSV
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        csv_file = output_dir / "kaggle_cleaned.csv"
        xlsx_file = output_dir / "kaggle_cleaned.xlsx"
        df.to_csv(csv_file, index=False)
        df.to_excel(xlsx_file, index=False)
        print(f"\n[OK] Saved Excel summary: {xlsx_file}")
        print(f"[OK] Saved CSV summary:   {csv_file}")

    print("=" * 70)
    print(f"Kaggle cleaning complete: {total_out} clean components extracted from {total_in} boxes.")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean Kaggle PCB Dataset")
    parser.add_argument("--images", default=r"dataset_split/train/images")
    parser.add_argument("--labels", default=r"dataset_split/train/labels")
    parser.add_argument("--output", default=r"cleaned_kaggle_data")
    parser.add_argument("--sam", default=r"c:\Userdata\antiiii\sam_vit_b.pth")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    clean_kaggle_dataset(Path(args.images), Path(args.labels), Path(args.output), Path(args.sam), args.limit)
