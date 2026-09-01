#!/usr/bin/env python3
"""
segment_pcb_wires.py

PCB Wire & Trace Segmentation Engine:
Extracts exact wire and copper trace pixel masks inside component/wire bounding boxes
using Lab background subtraction + Otsu thresholding + contour morphological refinement.
Outputs class-colored masks, visual overlays, and summary stats.
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path

def segment_wire_in_box(img_bgr, box, border_px=3, min_pixel_frac=0.01):
    """
    Extracts wire/trace pixel mask inside a box using Lab color space background subtraction.
    """
    x1, y1, x2, y2 = box
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < border_px * 2 + 2:
        return None

    # Convert to Lab color space for lighting-invariant color distance
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Estimate background color from outer border of bounding box
    border_mask = np.zeros(crop.shape[:2], dtype=bool)
    border_mask[:border_px, :] = True
    border_mask[-border_px:, :] = True
    border_mask[:, :border_px] = True
    border_mask[:, -border_px:] = True

    bg_pixels = lab[border_mask]
    bg_mean = bg_pixels.mean(axis=0)

    # Calculate distance of each pixel from border background mean
    dist = np.linalg.norm(lab - bg_mean, axis=2)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu thresholding to separate wire foreground from substrate background
    _, mask = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological refinement (close gaps, remove noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    fg_frac = np.count_nonzero(mask) / mask.size
    if fg_frac < min_pixel_frac or fg_frac > 0.95:
        return None

    return mask

def process_pcb_wire_segmentation(img_dir: Path, lbl_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    CLASS_COLORS = {
        'wire_power': (0, 0, 255),    # Red
        'wire_ground': (255, 0, 0),   # Blue
        'copper_trace': (0, 255, 255), # Yellow
        'component': (0, 255, 0)      # Green
    }

    print(f"Segmenting PCB wires & traces across {len(images)} images...")
    processed = 0

    for img_path in images:
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        color_mask = np.zeros_like(img)
        overlay = img.copy()

        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:5])
                x1 = max(0, int((xc - bw / 2.0) * w))
                y1 = max(0, int((yc - bh / 2.0) * h))
                x2 = min(w - 1, int((xc + bw / 2.0) * w))
                y2 = min(h - 1, int((yc + bh / 2.0) * h))

                box_mask = segment_wire_in_box(img, (x1, y1, x2, y2))
                bgr = list(CLASS_COLORS.values())[cid % len(CLASS_COLORS)]

                if box_mask is not None:
                    region = color_mask[y1:y2, x1:x2]
                    region[box_mask > 0] = bgr
                else:
                    color_mask[y1:y2, x1:x2] = bgr

                cv2.rectangle(overlay, (x1, y1), (x2, y2), bgr, 1)

        blended = cv2.addWeighted(overlay, 0.65, color_mask, 0.35, 0)
        cv2.imwrite(str(out_dir / f"{stem}_wire_mask.png"), color_mask)
        cv2.imwrite(str(out_dir / f"{stem}_wire_overlay.png"), blended)
        processed += 1

        if processed >= 10:
            break

    print(f"PCB Wire & Trace segmentation completed! Saved outputs in {out_dir}")

if __name__ == "__main__":
    img_d = Path(r"C:\Userdata\antiiii\dataset_split\train\images")
    lbl_d = Path(r"C:\Userdata\antiiii\dataset_split\train\labels")
    out_d = Path(r"C:\Userdata\antiiii\wire_trace_masks")
    process_pcb_wire_segmentation(img_d, lbl_d, out_d)
