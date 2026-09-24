#!/usr/bin/env python3
"""
run_sam.py

Pipeline 3 of 3: Runs SAM on PCB bounding boxes, extracts polygon points,
and saves the output to Excel (.xlsx) and CSV (.csv).
Also creates a LabelMe JSON and launches LabelMe for interactive viewing.
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
import pandas as pd
import torch
from segment_anything import sam_model_registry, SamPredictor

CLASS_MAP = {
    0: 'Cap1', 1: 'Cap2', 2: 'Cap3', 3: 'Cap4',
    4: 'MOSFET', 5: 'Mov', 6: 'Resistor', 7: 'Transformer'
}

PREFIX_MAP = {
    'Cap1': 'C', 'Cap2': 'C', 'Cap3': 'C', 'Cap4': 'C',
    'MOSFET': 'Q', 'Mov': 'D', 'Resistor': 'R', 'Transformer': 'T'
}

def extract_polygon_points(mask: np.ndarray, min_area: float = 15.0) -> List[List[float]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return []
    epsilon = 0.005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    return [[round(float(p[0][0]), 2), round(float(p[0][1]), 2)] for p in approx]

def run_sam(
    image_path: Path,
    labels_path: Path,
    sam_checkpoint: Path,
    output_dir: Path,
    open_labelme: bool = True
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("RUNNING SAM: POINT EXTRACTOR & EXCEL / CSV EXPORT")
    print("=" * 75)
    print(f"Image:  {image_path}")
    print(f"Labels: {labels_path}")
    print(f"Output: {output_dir}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    h, w = img.shape[:2]

    # Parse bounding boxes
    boxes = []
    with open(labels_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = int(round((xc - bw / 2.0) * w))
                y1 = int(round((yc - bh / 2.0) * h))
                x2 = int(round((xc + bw / 2.0) * w))
                y2 = int(round((yc + bh / 2.0) * h))
                boxes.append(([x1, y1, x2, y2], cid))

    print(f"\nParsed {len(boxes)} bounding boxes from label file.")

    # Initialize SAM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM ({sam_checkpoint}) on {device}...")
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print("SAM image embedding ready!\n")

    # Run inference & extract polygon points
    records = []
    shapes = []
    ref_counts = {}

    print("Extracted SAM Points:")
    print("-" * 75)

    for idx, (b, cid) in enumerate(boxes, 1):
        input_box = np.array(b)
        masks, scores, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
        mask = masks[0]
        conf = float(scores[0])

        pts = extract_polygon_points(mask)
        if not pts:
            pts = [[float(b[0]), float(b[1])], [float(b[2]), float(b[1])],
                   [float(b[2]), float(b[3])], [float(b[0]), float(b[3])]]

        class_name = CLASS_MAP.get(cid, f"Class_{cid}")
        prefix = PREFIX_MAP.get(class_name, "U")
        ref_counts[prefix] = ref_counts.get(prefix, 0) + 1
        ref_des = f"{prefix}{ref_counts[prefix]}"

        # Print points
        print(f"[{idx}] {ref_des:3s} ({class_name}): {len(pts)} points | Conf: {conf:.3f}")
        print(f"     Points: {pts}")

        # Record for Excel/CSV
        records.append({
            "image_name": image_path.name,
            "instance_id": idx,
            "ref_des": ref_des,
            "class_name": class_name,
            "class_id": cid,
            "box_x1": b[0],
            "box_y1": b[1],
            "box_x2": b[2],
            "box_y2": b[3],
            "confidence": round(conf, 3),
            "num_polygon_points": len(pts),
            "polygon_points_json": json.dumps(pts),
            "polygon_points_compact": "; ".join([f"({p[0]},{p[1]})" for p in pts])
        })

        shapes.append({
            "label": f"{ref_des}: {class_name}",
            "points": pts,
            "group_id": None,
            "description": f"SAM mask (conf: {conf:.2f})",
            "shape_type": "polygon",
            "flags": {},
            "mask": None
        })

    # Save to Excel (.xlsx) and CSV (.csv)
    df = pd.DataFrame(records)
    xlsx_path = output_dir / "sam_output_points.xlsx"
    csv_path = output_dir / "sam_output_points.csv"

    df.to_excel(xlsx_path, index=False)
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 75)
    print("OUTPUT SAVED:")
    print(f"  -> Excel Spreadsheet (.xlsx): {xlsx_path.resolve()}")
    print(f"  -> CSV File (.csv):           {csv_path.resolve()}")

    # Save LabelMe JSON & image copy
    dest_img = output_dir / image_path.name
    shutil.copy2(image_path, dest_img)

    json_path = output_dir / f"{image_path.stem}.json"
    labelme_payload = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_payload, f, indent=2)

    print(f"  -> LabelMe Annotation (.json):{json_path.resolve()}")

    # Launch LabelMe GUI
    if open_labelme:
        print("\nOpening in LabelMe...")
        try:
            subprocess.Popen([sys.executable, "-m", "labelme", str(json_path)])
            print("LabelMe launched successfully!")
        except Exception as e:
            print(f"Note on opening LabelMe: {e}")

    print("=" * 75)

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run SAM & Export Polygon Points to Excel / CSV")
    parser.add_argument("--image", default=str(BASE_DIR / "dataset_split" / "train" / "images" / "VID20210601143927-96_jpg.rf.36de73b8200ee94d0bd4679407c9cd40.jpg"))
    parser.add_argument("--labels", default=str(BASE_DIR / "dataset_split" / "train" / "labels" / "VID20210601143927-96_jpg.rf.36de73b8200ee94d0bd4679407c9cd40.txt"))
    parser.add_argument("--output", default=str(BASE_DIR / "sam_points_output"))
    parser.add_argument("--sam", default=str(BASE_DIR / "sam_vit_b.pth"))
    parser.add_argument("--no-labelme", action="store_true", help="Do not launch LabelMe GUI")
    args = parser.parse_args()

    run_sam(
        image_path=Path(args.image),
        labels_path=Path(args.labels),
        sam_checkpoint=Path(args.sam),
        output_dir=Path(args.output),
        open_labelme=not args.no_labelme
    )
