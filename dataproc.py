#!/usr/bin/env python3
"""
dataproc.py

PCB Component Detection & SAM Point Extractor Pipeline.
"""

import os
import sys
import glob
import json
import random
import urllib.request
import pandas as pd
import numpy as np
import cv2
import torch
from pathlib import Path

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/segment-anything.git"])
    from segment_anything import sam_model_registry, SamPredictor

try:
    import kagglehub
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub

KICAD_CLASSES = [
    'Capacitor_SMD',
    'Resistor_SMD',
    'Package_SO',           # IC Small Outline
    'Package_TO_SOT_SMD',   # Transistor / MOSFET
    'Diode_SMD',
    'Connector',
    'Inductor_SMD',
    'Button_Switch_SMD',
    'LED_SMD',
    'Transformer_SMD',
    'PCB_Defect',
    'Unknown_Component'
]

LABEL_SYNONYMS = {
    'Capacitor_SMD': ['c', 'cap', 'capacitor', 'capacitors', 'c_smd', 'c_tht', 'cap1', 'cap2', 'cap3', 'cap4'],
    'Resistor_SMD': ['r', 'res', 'resistor', 'resistors', 'r_smd', 'r_tht'],
    'Package_SO': ['ic', 'chip', 'integrated_circuit', 'soic', 'sop', 'qfp', 'qfn', 'dip', 'mcu'],
    'Package_TO_SOT_SMD': ['q', 'transistor', 'mosfet', 'fet', 'bjt', 'sot', 'sot23', 'to220', 'dopak', 'mov'],
    'Diode_SMD': ['d', 'diode', 'diodes', 'zener', 'schottky', 'tvs'],
    'Connector': ['conn', 'connector', 'connectors', 'header', 'plug', 'jack', 'usb', 'terminal'],
    'Inductor_SMD': ['l', 'ind', 'inductor', 'choke', 'coil'],
    'Button_Switch_SMD': ['sw', 'switch', 'button', 'btn', 'tactile', 'toggle'],
    'LED_SMD': ['led', 'leds', 'light_emitting_diode'],
    'Transformer_SMD': ['t', 'transformer', 'xfmr'],
    'PCB_Defect': ['defect', 'open', 'short', 'mousebite', 'spur', 'pin-hole', 'spurious_copper']
}

class LabelHomogenizer:
    def __init__(self):
        self.lookup = {}
        for std_name, synonyms in LABEL_SYNONYMS.items():
            for syn in synonyms:
                self.lookup[syn.lower()] = std_name
                
    def homogenize(self, raw_label):
        if isinstance(raw_label, int):
            return KICAD_CLASSES[raw_label] if raw_label < len(KICAD_CLASSES) else f"Class_{raw_label}"
            
        cleaned = str(raw_label).strip().lower()
        if cleaned in self.lookup:
            return self.lookup[cleaned]
            
        for syn, std_name in self.lookup.items():
            if syn in cleaned:
                return std_name
                
        return f"Unknown_{raw_label}"

homogenizer = LabelHomogenizer()

KAGGLE_DATASETS = {
    "fics_pcb": "ficslab/fics-pcb",
    "pku_pcb_defect": "akhatovar/pcb-defect-dataset",
    "deeppcb": "arnablaha/deeppcb"
}

SAM_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CHECKPOINT = Path(r"C:\Users\ANAGHA\sam_vit_b.pth")

def download_sam_weights():
    if not SAM_CHECKPOINT.exists():
        print(f"Downloading SAM pretrained weights to {SAM_CHECKPOINT}...")
        SAM_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAM_WEIGHTS_URL, str(SAM_CHECKPOINT))

def pull_kaggle_pcb_dataset(name="pku_pcb_defect"):
    slug = KAGGLE_DATASETS.get(name, name)
    try:
        path = kagglehub.dataset_download(slug)
        return Path(path)
    except Exception as e:
        print(f"Dataset Download Note: {e}")
        return None

def clean_bounding_boxes(boxes, img_w, img_h, min_box_size=10):
    cleaned_boxes = []
    for box, raw_lbl in boxes:
        x1, y1, x2, y2 = box
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))
        
        bw = x2 - x1
        bh = y2 - y1
        if bw >= min_box_size and bh >= min_box_size:
            cleaned_boxes.append(([x1, y1, x2, y2], raw_lbl))
    return cleaned_boxes

def extract_polygon_points(mask: np.ndarray, img_w: int, img_h: int):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon_lines = []
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 15:
            continue
            
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = approx.reshape(-1, 2)
        
        norm_pts = []
        for px, py in pts:
            norm_pts.extend([round(px / img_w, 6), round(py / img_h, 6)])
            
        if len(norm_pts) >= 6:
            polygon_lines.append(norm_pts)
            
    return polygon_lines

def export_to_labelme_json(image_name, img_w, img_h, polygon_records, output_json_path):
    shapes = []
    for rec in polygon_records:
        raw_lbl = rec.get('raw_label', rec.get('class_id'))
        kicad_name = homogenizer.homogenize(raw_lbl)
        
        pts_norm = rec['points']
        pixel_pts = []
        for i in range(0, len(pts_norm), 2):
            px = round(pts_norm[i] * img_w, 2)
            py = round(pts_norm[i+1] * img_h, 2)
            pixel_pts.append([px, py])
            
        shapes.append({
            "label": kicad_name,
            "points": pixel_pts,
            "group_id": None,
            "description": "SAM Polygon",
            "shape_type": "polygon",
            "flags": {}
        })
        
    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": img_h,
        "imageWidth": img_w
    }
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(labelme_data, f, indent=2)
        
    return output_json_path

def save_polygon_points_to_csv(extracted_records, output_csv_path):
    rows = []
    for record in extracted_records:
        raw_lbl = record.get('raw_label', record.get('class_id'))
        std_kicad_label = homogenizer.homogenize(raw_lbl)
        std_cid = KICAD_CLASSES.index(std_kicad_label) if std_kicad_label in KICAD_CLASSES else 11
        
        pts_pairs = [[record['points'][i], record['points'][i+1]] for i in range(0, len(record['points']), 2)]
        rows.append({
            'image_name': record['image_name'],
            'instance_id': record['instance_id'],
            'raw_label': raw_lbl,
            'homogenized_class_id': std_cid,
            'kicad_class_name': std_kicad_label,
            'num_points': len(pts_pairs),
            'points_json': json.dumps(pts_pairs),
            'points_yolo_str': f"{std_cid} " + " ".join(map(str, record['points']))
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv_path, index=False)
    return df

def run_sam_pipeline(dataset_dir: Path, output_dir: Path, augment: bool = True):
    download_sam_weights()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    out_seg_dir = output_dir / "labels_yolo_seg"
    out_labelme_dir = output_dir / "labels_labelme"
    out_vis_dir = output_dir / "visuals"
    out_seg_dir.mkdir(parents=True, exist_ok=True)
    out_labelme_dir.mkdir(parents=True, exist_ok=True)
    out_vis_dir.mkdir(parents=True, exist_ok=True)
    
    img_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    csv_records = []
    
    for img_path in img_files:
        stem = img_path.stem
        image = cv2.imread(str(img_path))
        if image is None:
            continue
            
        h, w = image.shape[:2]
        predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        txt_path = labels_dir / f"{stem}.txt"
        boxes = []
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        raw_cid = parts[0]
                        xc, yc, bw, bh = map(float, parts[1:5])
                        x1 = max(0, (xc - bw / 2) * w)
                        y1 = max(0, (yc - bh / 2) * h)
                        x2 = min(w, (xc + bw / 2) * w)
                        y2 = min(h, (yc + bh / 2) * h)
                        boxes.append(([x1, y1, x2, y2], raw_cid))
                        
        cleaned_boxes = clean_bounding_boxes(boxes, w, h)
        
        yolo_seg_rows = []
        img_records = []
        
        for inst_idx, (box, raw_lbl) in enumerate(cleaned_boxes):
            input_box = np.array(box)
            masks, _, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
            polygons = extract_polygon_points(masks[0], img_w=w, img_h=h)
            
            std_kicad_name = homogenizer.homogenize(raw_lbl)
            std_cid = KICAD_CLASSES.index(std_kicad_name) if std_kicad_name in KICAD_CLASSES else 11
            
            for pts in polygons:
                yolo_seg_rows.append(f"{std_cid} " + " ".join(map(str, pts)))
                rec = {
                    'image_name': img_path.name,
                    'instance_id': inst_idx,
                    'raw_label': raw_lbl,
                    'class_id': std_cid,
                    'points': pts
                }
                csv_records.append(rec)
                img_records.append(rec)
                
        with open(out_seg_dir / f"{stem}.txt", "w") as f:
            f.write("\n".join(yolo_seg_rows))
            
        export_to_labelme_json(img_path.name, w, h, img_records, out_labelme_dir / f"{stem}.json")
        
    if csv_records:
        save_polygon_points_to_csv(csv_records, output_dir / "polygon_points.csv")

if __name__ == "__main__":
    dataset_path = Path(r"C:\Userdata\antiiii\pcb_dataset")
    output_path = Path(r"C:\Userdata\antiiii\sam_output")
    (dataset_path / "images").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels").mkdir(parents=True, exist_ok=True)
    run_sam_pipeline(dataset_path, output_path, augment=True)
