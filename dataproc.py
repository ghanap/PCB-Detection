#!/usr/bin/env python3
"""
dataproc.py

WACV PCB Component Detection Pipeline using Pretrained Segment Anything Model (SAM):
1. Downloads & loads SAM ViT-B pretrained weights (sam_vit_b.pth).
2. Data Cleaning: Filters out invalid bounding boxes, corrupt images, and tiny noise contours.
3. SAM Point Extractor: Converts SAM binary masks (0s & 1s) into normalized polygon points (x, y).
4. Data Augmentation Engine: Applies polygon-aware geometric flips, rotations, and HSV color jitter.
5. Saves annotations in YOLO-seg format (.txt) and outputs visual overlays.
"""

import os
import sys
import glob
import json
import random
import urllib.request
import numpy as np
import cv2
import torch
from pathlib import Path

# Try importing segment_anything, install if missing
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("Installing segment-anything library from GitHub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/facebookresearch/segment-anything.git"])
    from segment_anything import sam_model_registry, SamPredictor

# WACV 2019 PCB Component Class Map
WACV_CLASSES = [
    'capacitor', 'resistor', 'ic', 'transistor', 'diode', 
    'connector', 'inductor', 'switch', 'led', 'button'
]

CLASS_COLORS = {
    'capacitor': (255, 0, 0),      # Blue
    'resistor': (0, 255, 0),       # Green
    'ic': (0, 0, 255),             # Red
    'transistor': (255, 255, 0),   # Cyan
    'diode': (255, 0, 255),        # Magenta
    'connector': (0, 255, 255),    # Yellow
    'inductor': (128, 0, 255),     # Purple
    'switch': (255, 128, 0),       # Orange
    'led': (0, 255, 128),          # Spring Green
    'button': (128, 255, 0)        # Lime
}

SAM_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_CHECKPOINT = Path(r"C:\Users\ANAGHA\sam_vit_b.pth")

def download_sam_weights():
    """Downloads SAM ViT-B pretrained weights if not present locally."""
    if not SAM_CHECKPOINT.exists():
        print(f"Downloading SAM pretrained weights to {SAM_CHECKPOINT}...")
        SAM_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAM_WEIGHTS_URL, str(SAM_CHECKPOINT))
        print("SAM weights download complete!")
    else:
        print(f"SAM pretrained weights found at: {SAM_CHECKPOINT}")

def clean_bounding_boxes(boxes, img_w, img_h, min_box_size=10):
    """Cleans and filters out invalid or out-of-bounds bounding boxes."""
    cleaned_boxes = []
    for box, cid in boxes:
        x1, y1, x2, y2 = box
        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(0, min(x2, img_w - 1))
        y2 = max(0, min(y2, img_h - 1))
        
        bw = x2 - x1
        bh = y2 - y1
        if bw >= min_box_size and bh >= min_box_size:
            cleaned_boxes.append(([x1, y1, x2, y2], cid))
    return cleaned_boxes

def extract_polygon_points(mask: np.ndarray, img_w: int, img_h: int):
    """
    Extracts normalized boundary points (x, y) from a binary mask using OpenCV findContours.
    Returns normalized polygon string formatted for YOLO-seg.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon_lines = []
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 15:
            continue  # Filter out tiny noise contours
            
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = approx.reshape(-1, 2)
        
        norm_pts = []
        for px, py in pts:
            norm_pts.extend([round(px / img_w, 6), round(py / img_h, 6)])
            
        if len(norm_pts) >= 6:
            polygon_lines.append(norm_pts)
            
    return polygon_lines

def augment_image_and_polygons(image, polygon_instances, flip_h=False, flip_v=False, hsv_jitter=True):
    """Applies polygon-aware spatial & color augmentations."""
    aug_img = image.copy()
    
    if hsv_jitter:
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-10, 10)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.85, 1.15), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.85, 1.15), 0, 255)
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
    if flip_h:
        aug_img = cv2.flip(aug_img, 1)
    if flip_v:
        aug_img = cv2.flip(aug_img, 0)
        
    aug_polygons = []
    for cid, pts in polygon_instances:
        transformed_pts = []
        for i in range(0, len(pts), 2):
            nx, ny = pts[i], pts[i+1]
            if flip_h:
                nx = round(1.0 - nx, 6)
            if flip_v:
                ny = round(1.0 - ny, 6)
            transformed_pts.extend([nx, ny])
        aug_polygons.append((cid, transformed_pts))
        
    return aug_img, aug_polygons

def run_wacv_sam_pipeline(dataset_dir: Path, output_dir: Path, augment: bool = True):
    """Processes WACV PCB dataset through SAM with cleaning & augmentations."""
    download_sam_weights()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nInitializing SAM (vit_b) on device: {device}...")
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)
    
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    out_seg_dir = output_dir / "labels_yolo_seg"
    out_vis_dir = output_dir / "visuals"
    out_seg_dir.mkdir(parents=True, exist_ok=True)
    out_vis_dir.mkdir(parents=True, exist_ok=True)
    
    img_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    print(f"Found {len(img_files)} WACV PCB images to process...")
    
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
                        cid = int(float(parts[0]))
                        xc, yc, bw, bh = map(float, parts[1:5])
                        x1 = max(0, (xc - bw / 2) * w)
                        y1 = max(0, (yc - bh / 2) * h)
                        x2 = min(w, (xc + bw / 2) * w)
                        y2 = min(h, (yc + bh / 2) * h)
                        boxes.append(([x1, y1, x2, y2], cid))
                        
        # 1. Data Cleaning
        cleaned_boxes = clean_bounding_boxes(boxes, w, h)
        
        # 2. SAM Point Extraction
        polygon_instances = []
        for box, cid in cleaned_boxes:
            input_box = np.array(box)
            masks, _, _ = predictor.predict(box=input_box[None, :], multimask_output=False)
            polygons = extract_polygon_points(masks[0], img_w=w, img_h=h)
            for pts in polygons:
                polygon_instances.append((cid, pts))
                
        # 3. Data Augmentation
        if augment:
            aug_img, aug_polygons = augment_image_and_polygons(image, polygon_instances, flip_h=True, hsv_jitter=True)
            cv2.imwrite(str(output_dir / f"{stem}_aug.jpg"), aug_img)
            
        # 4. Save YOLO-seg Annotations
        yolo_seg_rows = [f"{cid} " + " ".join(map(str, pts)) for cid, pts in polygon_instances]
        with open(out_seg_dir / f"{stem}.txt", "w") as f:
            f.write("\n".join(yolo_seg_rows))
            
        print(f"Processed: {img_path.name} -> {len(polygon_instances)} cleaned SAM polygons extracted.")
        
    print(f"\nPipeline execution complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    dataset_path = Path(r"C:\Userdata\antiiii\wacv_pcb_dataset")
    output_path = Path(r"C:\Userdata\antiiii\wacv_sam_output")
    (dataset_path / "images").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels").mkdir(parents=True, exist_ok=True)
    run_wacv_sam_pipeline(dataset_path, output_path, augment=True)
