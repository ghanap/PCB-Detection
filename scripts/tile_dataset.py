#!/usr/bin/env python3
"""
tile_dataset.py

Patch tiling engine for PCB images:
Tiles high-resolution PCB images into overlapping patches (default 640x640, 20% overlap / stride 512px),
re-mapping bounding boxes and polygon segmentation masks into patch coordinates.
Clips/discards shapes falling outside patch boundaries or below minimum area threshold.
"""

import os
import glob
import argparse
import numpy as np
import cv2
from pathlib import Path
import yaml

CLASS_NAMES = ['Cap1', 'Cap2', 'Cap3', 'Cap4', 'MOSFET', 'Mov', 'Resistor', 'Transformer']

def parse_yolo_label(lbl_path: Path, img_w: int, img_h: int):
    """
    Parses a YOLO txt label file (supports both 5-element bboxes and 7+ element polygons).
    Bounding Box format: class_id xc yc w h
    Polygon format: class_id x1 y1 x2 y2 ... xn yn
    Returns list of dicts: [{'class_id': int, 'points_px': [(x, y), ...]}]
    """
    objects = []
    if not lbl_path.exists():
        return objects

    with open(lbl_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cid = int(float(parts[0]))
            coords = list(map(float, parts[1:]))

            if len(coords) == 4:
                # Standard Bounding Box: xc, yc, w, h
                xc, yc, w, h = coords
                x1 = (xc - w / 2.0) * img_w
                y1 = (yc - h / 2.0) * img_h
                x2 = (xc + w / 2.0) * img_w
                y2 = (yc + h / 2.0) * img_h
                pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            elif len(coords) >= 6:
                # Polygon Segmentation format: x1, y1, x2, y2, ...
                pts = []
                for i in range(0, len(coords), 2):
                    px = coords[i] * img_w
                    py = coords[i+1] * img_h
                    pts.append((px, py))
            else:
                continue

            objects.append({'class_id': cid, 'points_px': pts})

    return objects

def clip_polygon_to_patch(pts: list, patch_x: int, patch_y: int, patch_w: int, patch_h: int, min_area: float = 10.0):
    """
    Clips polygon/bbox points to the patch bounds and re-maps to patch local coordinates [0, 1].
    Returns list of normalized (x, y) tuples if valid polygon remains, else empty list.
    """
    poly_abs = np.array(pts, dtype=np.float32)

    # Re-map relative to patch origin
    poly_patch = poly_abs - np.array([patch_x, patch_y], dtype=np.float32)

    mask = np.zeros((patch_h, patch_w), dtype=np.uint8)
    poly_int = poly_patch.astype(np.int32)
    cv2.fillPoly(mask, [poly_int], 255)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clipped_polys = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts_clipped = approx.reshape(-1, 2)

        if len(pts_clipped) >= 3:
            norm_pts = []
            for px, py in pts_clipped:
                norm_pts.extend([round(float(px) / patch_w, 6), round(float(py) / patch_h, 6)])
            clipped_polys.append(norm_pts)

    return clipped_polys

def tile_image(img_path: Path, lbl_path: Path, out_img_dir: Path, out_lbl_dir: Path,
               patch_size: int = 640, overlap: float = 0.2, min_area: float = 10.0,
               keep_empty: bool = False):
    """Tiles a single image and its corresponding label file."""
    image = cv2.imread(str(img_path))
    if image is None:
        return 0

    img_h, img_w = image.shape[:2]
    stride = int(patch_size * (1.0 - overlap))

    objects = parse_yolo_label(lbl_path, img_w, img_h)

    stem = img_path.stem
    x_steps = list(range(0, max(1, img_w - patch_size + 1), stride))
    y_steps = list(range(0, max(1, img_h - patch_size + 1), stride))

    if x_steps[-1] + patch_size < img_w:
        x_steps.append(img_w - patch_size)
    if y_steps[-1] + patch_size < img_h:
        y_steps.append(img_h - patch_size)

    patch_count = 0

    for py in y_steps:
        for px in x_steps:
            patch = image[py:py+patch_size, px:px+patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                patch = cv2.copyMakeBorder(
                    patch, 0, patch_size - patch.shape[0], 0, patch_size - patch.shape[1],
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )

            patch_yolo_lines = []

            for obj in objects:
                cid = obj['class_id']
                clipped_polys = clip_polygon_to_patch(
                    obj['points_px'], px, py, patch_size, patch_size, min_area
                )
                for norm_pts in clipped_polys:
                    patch_yolo_lines.append(f"{cid} " + " ".join(map(str, norm_pts)))

            if patch_yolo_lines or keep_empty:
                patch_name = f"{stem}_tile_{px}_{py}"
                cv2.imwrite(str(out_img_dir / f"{patch_name}.jpg"), patch)

                with open(out_lbl_dir / f"{patch_name}.txt", "w") as f:
                    f.write("\n".join(patch_yolo_lines))

                patch_count += 1

    return patch_count

def process_tiling(dataset_dir: Path, output_dir: Path, patch_size: int = 640,
                   overlap: float = 0.2, min_area: float = 10.0):
    """Processes tiling for all splits (train/val/test)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total_tiles = 0
    for split in ['train', 'val', 'test']:
        split_src = dataset_dir / split
        if not split_src.exists():
            continue

        out_img_dir = output_dir / split / "images"
        out_lbl_dir = output_dir / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_dir = split_src / "images"
        lbl_dir = split_src / "labels_seg"
        if not lbl_dir.exists() or not list(lbl_dir.glob("*.txt")):
            lbl_dir = split_src / "labels"

        images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        print(f"Tiling '{split}' set ({len(images)} images) from '{lbl_dir.name}' into {patch_size}x{patch_size} patches (overlap={overlap})...")

        split_tiles = 0
        for img_path in images:
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            n_tiles = tile_image(
                img_path, lbl_path, out_img_dir, out_lbl_dir,
                patch_size=patch_size, overlap=overlap, min_area=min_area
            )
            split_tiles += n_tiles

        print(f"  [{split}] Created {split_tiles} patch tiles.")
        total_tiles += split_tiles

    yaml_data = {
        'path': str(output_dir.resolve()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(CLASS_NAMES),
        'names': CLASS_NAMES
    }

    yaml_path = output_dir / "fics_pcb_tiled.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)

    print(f"\nTiling complete! Total patches: {total_tiles}. Configuration saved to: {yaml_path}")

def main():
    parser = argparse.ArgumentParser(description="Tile PCB dataset into overlapping patches.")
    parser.add_argument("--dataset_dir", type=str, default=r"C:\Userdata\antiiii\dataset_split",
                        help="Input dataset split directory")
    parser.add_argument("--output_dir", type=str, default=r"C:\Userdata\antiiii\dataset_tiled",
                        help="Output tiled dataset directory")
    parser.add_argument("--patch_size", type=int, default=640, help="Patch size in pixels (default 640)")
    parser.add_argument("--overlap", type=float, default=0.20, help="Overlap fraction (default 0.20)")
    parser.add_argument("--min_area", type=float, default=10.0, help="Minimum contour area in pixels")
    args = parser.parse_args()

    process_tiling(
        Path(args.dataset_dir), Path(args.output_dir),
        patch_size=args.patch_size, overlap=args.overlap, min_area=args.min_area
    )

if __name__ == "__main__":
    main()
