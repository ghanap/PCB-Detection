#!/usr/bin/env python3
"""
wacv_sam_pipeline.py

WACV PCB Component Detection Pipeline using Pretrained Segment Anything Model (SAM).
1. Loads/Downloads SAM ViT-B pretrained weights (sam_vit_b.pth).
2. Parses WACV PCB dataset bounding boxes.
3. Passes bounding boxes into SAM as box prompts to extract 2D object masks.
4. Converts binary masks into normalized polygon (x, y) boundary points via OpenCV contours.
5. Saves annotations in YOLO-seg format (.txt) and creates visual overlay previews.
"""

import os
import sys
import glob
import json
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

def extract_polygon_points(mask: np.ndarray, img_w: int, img_h: int):
    """
    Extracts normalized boundary points (x, y) from a binary mask using OpenCV findContours.
    Returns normalized polygon string formatted for YOLO-seg.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon_lines = []
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 10:
            continue  # Skip tiny noise points
            
        # Simplify contour to smooth polygon boundary
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = approx.reshape(-1, 2)
        
        # Normalize points between 0.0 and 1.0 for YOLO
        norm_pts = []
        for px, py in pts:
            norm_pts.extend([round(px / img_w, 6), round(py / img_h, 6)])
            
        if len(norm_pts) >= 6:  # Polygon requires at least 3 points (6 floats)
            polygon_lines.append(norm_pts)
            
    return polygon_lines

def run_wacv_sam_pipeline(dataset_dir: Path, output_dir: Path):
    """
    Processes images and bounding boxes from WACV dataset through SAM
    to generate YOLO-seg polygon points.
    """
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
        json_path = labels_dir / f"{stem}.json"
        
        boxes = []
        # Parse bounding boxes (support both YOLO txt format and JSON format)
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
                        
        yolo_seg_rows = []
        overlay = image.copy()
        
        for box, cid in boxes:
            input_box = np.array(box)
            masks, scores, _ = predictor.predict(
                box=input_box[None, :],
                multimask_output=False
            )
            mask = masks[0]
            
            # Extract list of points
            polygons = extract_polygon_points(mask, img_w=w, img_h=h)
            for pts in polygons:
                row_str = f"{cid} " + " ".join(map(str, pts))
                yolo_seg_rows.append(row_str)
                
            # Draw visual preview overlay
            lbl_name = WACV_CLASSES[cid] if cid < len(WACV_CLASSES) else str(cid)
            color = CLASS_COLORS.get(lbl_name, (0, 255, 0))
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
            cv2.rectangle(overlay, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1)
            cv2.putText(overlay, lbl_name, (int(box[0]), int(box[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        # Save output YOLO-seg label text file
        out_txt = out_seg_dir / f"{stem}.txt"
        with open(out_txt, "w") as f:
            f.write("\n".join(yolo_seg_rows))
            
        # Save visual overlay image
        cv2.imwrite(str(out_vis_dir / f"{stem}_sam_polygons.jpg"), overlay)
        print(f"Processed: {img_path.name} -> {len(yolo_seg_rows)} polygon instances extracted.")
        
    print(f"\nPipeline execution complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    dataset_path = Path(r"C:\Userdata\antiiii\wacv_pcb_dataset")
    output_path = Path(r"C:\Userdata\antiiii\wacv_sam_output")
    
    # Create sample placeholder directory structure if dataset not populated yet
    (dataset_path / "images").mkdir(parents=True, exist_ok=True)
    (dataset_path / "labels").mkdir(parents=True, exist_ok=True)
    
    run_wacv_sam_pipeline(dataset_path, output_path)
