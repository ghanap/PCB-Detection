#!/usr/bin/env python3
"""
clean_wacv.py

Pipeline for WACV 2019 PCB Dataset:
- Ingests Pascal VOC XML annotations and board images.
- Filters out non-components ('text', 'pads').
- Cleans and clamps component bounding boxes.
- Prompts Meta's SAM (Segment Anything Model) to extract polygon points [(x, y), ...].
- Maps classes to KiCad Library Convention (KLC) footprints and Reference Designators.
- Progressively exports results to Excel (.xlsx), CSV (.csv), and LabelMe JSON (.json).
"""

import os
import sys
import json
import shutil
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Any

import cv2
import numpy as np
import pandas as pd
import torch
from segment_anything import sam_model_registry, SamPredictor

WACV_CLASSES = {
    'capacitor': {'kicad': 'Capacitor_SMD',      'footprint': 'Capacitor_SMD:C_0805_2012Metric',                       'ref_prefix': 'C'},
    'electrolytic': {'kicad': 'Capacitor_THT',   'footprint': 'Capacitor_THT:CP_Radial_D6.3mm_P2.50mm',                 'ref_prefix': 'C'},
    'resistor':   {'kicad': 'Resistor_SMD',       'footprint': 'Resistor_SMD:R_0805_2012Metric',                        'ref_prefix': 'R'},
    'ic':         {'kicad': 'Package_SO',          'footprint': 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm',                    'ref_prefix': 'U'},
    'transistor': {'kicad': 'Package_TO_SOT_SMD', 'footprint': 'Package_TO_SOT_SMD:SOT-23',                             'ref_prefix': 'Q'},
    'diode':      {'kicad': 'Diode_SMD',          'footprint': 'Diode_SMD:D_SOD-123',                                    'ref_prefix': 'D'},
    'connector':  {'kicad': 'Connector',          'footprint': 'Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vert', 'ref_prefix': 'J'},
    'inductor':   {'kicad': 'Inductor_SMD',       'footprint': 'Inductor_SMD:L_0805_2012Metric',                        'ref_prefix': 'L'},
    'switch':     {'kicad': 'Button_Switch_SMD',  'footprint': 'Button_Switch_SMD:SW_Push_SPST_NO_Alps_SKRK',            'ref_prefix': 'SW'},
    'button':     {'kicad': 'Button_Switch_SMD',  'footprint': 'Button_Switch_SMD:SW_Push_SPST_NO_Alps_SKRK',            'ref_prefix': 'SW'},
    'led':        {'kicad': 'LED_SMD',            'footprint': 'LED_SMD:LED_0805_2012Metric',                            'ref_prefix': 'D'},
    'clock':      {'kicad': 'Crystal',            'footprint': 'Crystal:Crystal_SMD_3225-4Pin_3.2x2.5mm',               'ref_prefix': 'Y'},
    'fuse':       {'kicad': 'Fuse',               'footprint': 'Fuse:Fuse_1206_3216Metric',                             'ref_prefix': 'F'},
    'transformer':{'kicad': 'Transformer_SMD',    'footprint': 'Transformer_SMD:Transformer_Bourns_SRF0703',            'ref_prefix': 'T'},
    'component':  {'kicad': 'Generic_Component',  'footprint': 'Package_SO:SOIC-8_3.9x4.9mm_P1.27mm',                    'ref_prefix': 'U'}
}

IGNORED_LABELS = {'text', 'pads', 'pins', 'unknown', 'test'}

def clean_box(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int, min_size: int = 8) -> Tuple[bool, List[int]]:
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    x1 = max(0, min(int(round(x1)), img_w - 1))
    y1 = max(0, min(int(round(y1)), img_h - 1))
    x2 = max(0, min(int(round(x2)), img_w))
    y2 = max(0, min(int(round(y2)), img_h))
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
    return [[round(float(p[0][0]), 2), round(float(p[0][1]), 2)] for p in approx]

def resolve_label(raw_name: str) -> Tuple[str, Dict[str, str]]:
    cleaned = raw_name.strip().strip('"').lower()
    main_type = cleaned.split()[0] if cleaned else "component"
    main_type = main_type.strip('"')

    for key, info in WACV_CLASSES.items():
        if key in main_type:
            return key, info
    return 'component', WACV_CLASSES['component']

def process_single_board(
    board_dir: Path,
    predictor: SamPredictor,
    output_dir: Path,
    max_components: int = None
) -> List[Dict[str, Any]]:
    board_name = board_dir.name

    xml_files = list(board_dir.glob("*.xml"))
    if not xml_files:
        return []
    xml_path = xml_files[0]

    img_files = list(board_dir.glob("*.jpg")) + list(board_dir.glob("*.png"))
    if not img_files:
        return []
    jpg_matches = [p for p in img_files if p.suffix.lower() == '.jpg']
    img_path = jpg_matches[0] if jpg_matches else img_files[0]

    img = cv2.imread(str(img_path))
    if img is None:
        return []
    h, w = img.shape[:2]

    tree = ET.parse(xml_path)
    root = tree.getroot()

    parsed_boxes = []
    for obj in root.findall('object'):
        name_tag = obj.find('name')
        raw_name = name_tag.text if name_tag is not None else ""
        first_word = raw_name.strip().strip('"').lower().split()[0] if raw_name else ""

        if first_word in IGNORED_LABELS:
            continue

        bnd = obj.find('bndbox')
        if bnd is None:
            continue

        xmin = float(bnd.find('xmin').text)
        ymin = float(bnd.find('ymin').text)
        xmax = float(bnd.find('xmax').text)
        ymax = float(bnd.find('ymax').text)

        ok, box = clean_box(xmin, ymin, xmax, ymax, w, h)
        if ok:
            norm_type, info = resolve_label(raw_name)
            parsed_boxes.append((box, norm_type, info, raw_name))

    if not parsed_boxes:
        return []

    if max_components:
        parsed_boxes = parsed_boxes[:max_components]

    print(f"Board '{board_name}': segmenting {len(parsed_boxes)} components...")
    predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    records = []
    shapes = []
    ref_counts = {}

    for idx, (box, norm_type, info, raw_name) in enumerate(parsed_boxes, 1):
        masks, scores, _ = predictor.predict(box=np.array(box)[None, :], multimask_output=False)
        conf = float(scores[0])
        pts = extract_polygon(masks[0])
        if not pts:
            pts = [[float(box[0]), float(box[1])], [float(box[2]), float(box[1])],
                   [float(box[2]), float(box[3])], [float(box[0]), float(box[3])]]

        pfx = info['ref_prefix']
        ref_counts[pfx] = ref_counts.get(pfx, 0) + 1
        ref_des = f"{pfx}{ref_counts[pfx]}"

        records.append({
            "board_name": board_name,
            "instance_id": idx,
            "ref_des": ref_des,
            "wacv_type": norm_type,
            "raw_annotation": raw_name,
            "kicad_class": info['kicad'],
            "kicad_footprint": info['footprint'],
            "box_x1": box[0], "box_y1": box[1],
            "box_x2": box[2], "box_y2": box[3],
            "confidence": round(conf, 3),
            "num_polygon_points": len(pts),
            "polygon_points_compact": "; ".join([f"({p[0]},{p[1]})" for p in pts]),
            "polygon_points_json": json.dumps(pts)
        })

        shapes.append({
            "label": f"{ref_des}: {norm_type.upper()}",
            "points": pts,
            "shape_type": "polygon"
        })

    # Save copy of image and LabelMe JSON
    dest_img = output_dir / img_path.name
    if not dest_img.exists():
        shutil.copy2(img_path, dest_img)

    json_path = output_dir / f"{img_path.stem}.json"
    labelme_payload = {
        "version": "5.5.0",
        "flags": {},
        "shapes": shapes,
        "imagePath": img_path.name,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_payload, f, indent=2)

    return records

def run_wacv_pipeline(
    source_dir: Path,
    sam_checkpoint: Path,
    output_dir: Path,
    limit_boards: int = None,
    limit_components: int = None
):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "wacv_all_boards_sam_points.csv"
    xlsx_path = output_dir / "wacv_all_boards_sam_points.xlsx"

    # Identify if source is single board or parent folder containing multiple boards
    if (source_dir / "ArduinoMega_Top").exists() or len(list(source_dir.glob("*.xml"))) == 0:
        board_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir() and list(d.glob("*.xml"))])
    else:
        board_dirs = [source_dir]

    if limit_boards:
        board_dirs = board_dirs[:limit_boards]

    print(f"Processing {len(board_dirs)} WACV boards...")

    # Load SAM once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM ({sam_checkpoint}) on {device}...")
    sam = sam_model_registry["vit_b"](checkpoint=str(sam_checkpoint))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    all_records = []
    for idx, b_dir in enumerate(board_dirs, 1):
        print(f"[{idx}/{len(board_dirs)}] Processing board: {b_dir.name}")
        records = process_single_board(b_dir, predictor, output_dir, max_components=limit_components)
        all_records.extend(records)

        # Progressively save spreadsheets
        if all_records:
            df = pd.DataFrame(all_records)
            df.to_csv(csv_path, index=False)
            df.to_excel(xlsx_path, index=False)

    print(f"Completed WACV extraction: {len(all_records)} total components across {len(board_dirs)} boards.")
    print(f"Saved: {csv_path}")

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent
    DEFAULT_WACV_DIR = BASE_DIR / "wacv_data" / "pcb_wacv_2019"
    DEFAULT_SAM = BASE_DIR / "sam_vit_b.pth"
    DEFAULT_OUT = BASE_DIR / "wacv_sam_output"

    parser = argparse.ArgumentParser(description="Clean WACV PCB & Extract SAM Polygon Points")
    parser.add_argument("--source", default=str(DEFAULT_WACV_DIR), help="Path to board folder or pcb_wacv_2019 directory")
    parser.add_argument("--sam", default=str(DEFAULT_SAM), help="Path to SAM ViT-B checkpoint")
    parser.add_argument("--output", default=str(DEFAULT_OUT), help="Output directory")
    parser.add_argument("--limit-boards", type=int, default=None, help="Limit number of boards to process")
    parser.add_argument("--limit-components", type=int, default=None, help="Limit components per board")
    args = parser.parse_args()

    run_wacv_pipeline(
        source_dir=Path(args.source),
        sam_checkpoint=Path(args.sam),
        output_dir=Path(args.output),
        limit_boards=args.limit_boards,
        limit_components=args.limit_components
    )
