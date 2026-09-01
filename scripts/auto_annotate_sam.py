#!/usr/bin/env python3
"""
auto_annotate_sam.py

Fast SAM auto-mask generator:
Uses SAM (Segment Anything Model) in box-prompt mode with PyTorch CPU multi-threading
and instant resume capability (skips already-completed images).
"""

import os
import glob
import json
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamPredictor

CLASS_NAMES = ['Cap1', 'Cap2', 'Cap3', 'Cap4', 'MOSFET', 'Mov', 'Resistor', 'Transformer']

CLASS_COLORS = {
    'Cap1': (255, 255, 0),
    'Cap2': (255, 128, 0),
    'Cap3': (255, 0, 128),
    'Cap4': (255, 0, 255),
    'MOSFET': (50, 50, 255),
    'Mov': (0, 140, 255),
    'Resistor': (0, 230, 255),
    'Transformer': (77, 230, 0)
}

def parse_boxes_from_json(json_path: Path):
    boxes = []
    if not json_path.exists():
        return boxes
    with open(json_path, 'r') as f:
        data = json.load(f)
    for shape in data.get('shapes', []):
        lbl = shape.get('label', '')
        pts = shape.get('points', [])
        if len(pts) >= 2:
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            cid = CLASS_NAMES.index(lbl) if lbl in CLASS_NAMES else 0
            box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            boxes.append((box, lbl, cid))
    return boxes

def parse_boxes_from_yolo(txt_path: Path, img_w: int, img_h: int):
    boxes = []
    if not txt_path.exists():
        return boxes
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:5])
                x1 = (xc - w / 2) * img_w
                y1 = (yc - h / 2) * img_h
                x2 = (xc + w / 2) * img_w
                y2 = (yc + h / 2) * img_h
                lbl = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid)
                boxes.append(([x1, y1, x2, y2], lbl, cid))
    return boxes

def process_split(split_dir: Path, predictor: SamPredictor, save_vis: bool = False):
    img_dir = split_dir / "images"
    json_dir = split_dir / "json_labels"
    lbl_dir = split_dir / "labels"
    out_seg_dir = split_dir / "labels_seg"
    out_vis_dir = split_dir / "visuals_sam"

    out_seg_dir.mkdir(parents=True, exist_ok=True)
    if save_vis:
        out_vis_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    total_imgs = len(img_files)

    # Resume check
    to_process = []
    already_done = 0
    for img_path in img_files:
        stem = img_path.stem
        out_seg_path = out_seg_dir / f"{stem}.txt"
        if out_seg_path.exists() and out_seg_path.stat().st_size > 0:
            already_done += 1
        else:
            to_process.append(img_path)

    print(f"\n[{split_dir.name.upper()}] Total: {total_imgs} | Already Done: {already_done} | Remaining: {len(to_process)}")

    if not to_process:
        return

    processed_count = 0
    polys_count = 0

    with torch.no_grad():
        for i, img_path in enumerate(to_process):
            stem = img_path.stem
            out_seg_path = out_seg_dir / f"{stem}.txt"

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            h, w = image.shape[:2]

            boxes_with_labels = parse_boxes_from_json(json_dir / f"{stem}.json")
            if not boxes_with_labels:
                boxes_with_labels = parse_boxes_from_yolo(lbl_dir / f"{stem}.txt", w, h)

            if not boxes_with_labels:
                continue

            predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            overlay = image.copy() if save_vis else None
            yolo_seg_lines = []

            for box, lbl, cid in boxes_with_labels:
                input_box = np.array(box)
                masks, scores, _ = predictor.predict(
                    box=input_box[None, :],
                    multimask_output=False
                )
                mask = masks[0].astype(np.uint8)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area < 10:
                        continue

                    epsilon = 0.005 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, epsilon, True)
                    pts = approx.reshape(-1, 2)

                    norm_pts = []
                    for px, py in pts:
                        norm_pts.extend([round(px / w, 6), round(py / h, 6)])

                    if len(norm_pts) >= 6:
                        yolo_seg_lines.append(f"{cid} " + " ".join(map(str, norm_pts)))
                        polys_count += 1

                if save_vis and overlay is not None:
                    color = CLASS_COLORS.get(lbl, (0, 255, 0))
                    color_mask = np.zeros_like(image, dtype=np.uint8)
                    color_mask[mask == 1] = color
                    overlay[mask == 1] = cv2.addWeighted(overlay[mask == 1], 0.4, color_mask[mask == 1], 0.6, 0)
                    cv2.drawContours(overlay, contours, -1, color, 2)

            with open(out_seg_path, "w") as f:
                f.write("\n".join(yolo_seg_lines))

            if save_vis and overlay is not None:
                cv2.imwrite(str(out_vis_dir / f"{stem}_sam.jpg"), overlay)

            processed_count += 1
            if processed_count % 25 == 0 or processed_count == len(to_process):
                print(f"  [{split_dir.name}] Progress: {already_done + processed_count}/{total_imgs} images done ({polys_count} polygons)...")

    print(f"[{split_dir.name.upper()}] Split processing finished!")

def main():
    parser = argparse.ArgumentParser(description="Accelerated SAM auto-mask generator.")
    parser.add_argument("--dataset_dir", type=str, default=r"C:\Userdata\antiiii\dataset_split",
                        help="Path to split dataset directory")
    parser.add_argument("--checkpoint", type=str, default=r"C:\Users\ANAGHA\sam_vit_b.pth",
                        help="Path to SAM weights (.pth)")
    parser.add_argument("--model_type", type=str, default="vit_b", help="SAM model type")
    parser.add_argument("--threads", type=int, default=8, help="PyTorch CPU threads (default 8)")
    parser.add_argument("--no_vis", action="store_true", help="Disable visual overlays")
    args = parser.parse_args()

    # Configure PyTorch CPU multi-threading
    torch.set_num_threads(args.threads)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading SAM model on device: {device} with PyTorch threads={args.threads}...")

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    dataset_path = Path(args.dataset_dir)
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        if split_dir.exists():
            process_split(split_dir, predictor, save_vis=not args.no_vis)

if __name__ == "__main__":
    main()
