#!/usr/bin/env python3
"""
test_pcb_model.py

Inference & Testing Script:
Runs trained YOLOv11-seg model on PCB images to predict component locations,
color-coded pixel segmentation masks, confidence scores, and class labels.
Saves rendered visual prediction outputs.
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

CLASS_NAMES = ['Cap1', 'Cap2', 'Cap3', 'Cap4', 'MOSFET', 'Mov', 'Resistor', 'Transformer']
CLASS_COLORS = [
    (255, 255, 0),   # Cap1 (Cyan)
    (255, 128, 0),   # Cap2 (Sky Blue)
    (255, 0, 128),   # Cap3 (Purple)
    (255, 0, 255),   # Cap4 (Magenta)
    (50, 50, 255),   # MOSFET (Red)
    (0, 140, 255),   # Mov (Orange)
    (0, 230, 255),   # Resistor (Yellow)
    (77, 230, 0)     # Transformer (Green)
]

def run_model_inference(weights_path: str, test_img_dir: Path, out_path: Path):
    print(f"Loading YOLOv11-seg model from {weights_path}...")
    model = YOLO(weights_path)

    img_files = sorted(list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png")))
    if not img_files:
        print(f"No test images found in {test_img_dir}")
        return

    rendered_list = []
    print(f"Running inference on test PCB images...")

    for img_path in img_files:
        results = model(str(img_path), imgsz=640, conf=0.25, verbose=False)
        res = results[0]

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        overlay = img.copy()
        color_mask = np.zeros_like(img)

        # Process predicted masks and bounding boxes
        if res.masks is not None and res.boxes is not None:
            masks_xy = res.masks.xy
            boxes_cls = res.boxes.cls.cpu().numpy()
            boxes_conf = res.boxes.conf.cpu().numpy()

            for pts_norm, cid, conf in zip(masks_xy, boxes_cls, boxes_conf):
                cid = int(cid)
                cname = CLASS_NAMES[cid % len(CLASS_NAMES)]
                color = CLASS_COLORS[cid % len(CLASS_COLORS)]

                pts = pts_norm.astype(np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(color_mask, [pts], color)
                    cv2.polylines(overlay, [pts], True, color, 2)

                    # Text label with confidence score
                    x, y = pts[0]
                    lbl_text = f"{cname} {conf*100:.0f}%"
                    cv2.putText(overlay, lbl_text, (x, max(y - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
                    cv2.putText(overlay, lbl_text, (x, max(y - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        blended = cv2.addWeighted(overlay, 0.65, color_mask, 0.35, 0)
        resized = cv2.resize(blended, (640, 640))
        rendered_list.append(resized)

        if len(rendered_list) >= 4:
            break

    if len(rendered_list) >= 2:
        if len(rendered_list) >= 4:
            row1 = np.hstack([rendered_list[0], rendered_list[1]])
            row2 = np.hstack([rendered_list[2], rendered_list[3]])
            grid = np.vstack([row1, row2])
        else:
            grid = np.hstack([rendered_list[0], rendered_list[1]])

        cv2.imwrite(str(out_path), grid)
        print(f"Saved prediction test preview to: {out_path}")

if __name__ == "__main__":
    weights = "yolo11n-seg.pt"  # Pretrained checkpoint or fine-tuned best.pt
    test_dir = Path(r"C:\Userdata\antiiii\dataset_split\test\images")
    art_path = Path(r"C:\Users\ANAGHA\.gemini\antigravity\brain\a18c0ee1-53bb-4527-a3d0-d12dbe11c8f0\pcb_model_prediction_test.png")
    run_model_inference(weights, test_dir, art_path)
