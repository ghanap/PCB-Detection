#!/usr/bin/env python3
"""
render-masks.py

Takes SAM polygon outputs (LabelMe JSON format) and renders filled,
semi-transparent colored mask overlays on the original PCB images.

Input: directory containing image files and matching .json files
       (as produced by run_sam.py or clean-and-segment-pipeline.py)
Output: <stem>-masked.png for each image, saved to --output dir

Usage:
  python render-masks.py --input path/to/sam-results --output path/to/masked-output
  python render-masks.py --input path/to/sam-results   # saves next to originals
"""

import os
import json
import argparse
from pathlib import Path

import cv2
import numpy as np


# one color per class, BGR format
CLASS_COLORS = {
    "Cap1":        (30, 180, 255),   # orange
    "Cap2":        (255, 180, 30),   # blue
    "Cap3":        (50, 220, 50),    # green
    "Cap4":        (180, 50, 255),   # magenta
    "MOSFET":      (0, 0, 220),      # red
    "Mov":         (220, 220, 0),    # cyan
    "Resistor":    (0, 220, 220),    # yellow
    "Transformer": (200, 100, 255),  # pink
    # WACV classes
    "capacitor":   (30, 180, 255),
    "resistor":    (0, 220, 220),
    "ic":          (255, 100, 100),
    "connector":   (100, 255, 100),
    "inductor":    (255, 200, 100),
    "diode":       (100, 100, 255),
    "led":         (0, 255, 255),
    "component":   (180, 180, 180),
    "transistor":  (0, 0, 220),
    "crystal":     (255, 255, 100),
    "fuse":        (100, 255, 255),
    "switch":      (200, 100, 200),
    "button":      (150, 200, 100),
    "relay":       (100, 200, 200),
}

# fallback palette for unknown classes
FALLBACK_COLORS = [
    (255, 127, 0), (0, 127, 255), (127, 255, 0),
    (255, 0, 127), (0, 255, 127), (127, 0, 255),
    (200, 200, 50), (50, 200, 200), (200, 50, 200),
]


def get_class_from_label(label: str) -> str:
    """Extract class name from LabelMe label like 'C1: Cap1' or 'R3: Resistor'."""
    if ":" in label:
        return label.split(":", 1)[1].strip()
    return label.strip()


def get_color(class_name: str, fallback_idx: int) -> tuple:
    """Get BGR color for a class, falling back to palette rotation."""
    if class_name in CLASS_COLORS:
        return CLASS_COLORS[class_name]
    return FALLBACK_COLORS[fallback_idx % len(FALLBACK_COLORS)]


def render_masked_image(image: np.ndarray, shapes: list, alpha: float = 0.45) -> np.ndarray:
    """
    Render filled polygon masks over the image.

    Args:
        image: original BGR image
        shapes: list of LabelMe shape dicts with 'label' and 'points'
        alpha: transparency of the overlay (0 = invisible, 1 = opaque)

    Returns:
        BGR image with colored mask overlays and labels
    """
    overlay = image.copy()
    output = image.copy()
    fallback_idx = 0

    for shape in shapes:
        label = shape.get("label", "unknown")
        points = shape.get("points", [])
        if len(points) < 3:
            continue

        class_name = get_class_from_label(label)
        color = get_color(class_name, fallback_idx)
        if class_name not in CLASS_COLORS:
            fallback_idx += 1

        pts = np.array(points, dtype=np.int32)

        # filled polygon on the overlay
        cv2.fillPoly(overlay, [pts], color)

        # thin border on the output for definition
        cv2.polylines(output, [pts], isClosed=True, color=color, thickness=2)

    # blend the filled overlay with the original
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    # draw labels on top (after blending so they stay crisp)
    for shape in shapes:
        label = shape.get("label", "unknown")
        points = shape.get("points", [])
        if len(points) < 3:
            continue

        class_name = get_class_from_label(label)
        color = get_color(class_name, 0)

        pts = np.array(points, dtype=np.int32)
        x, y = pts[:, 0].min(), pts[:, 1].min() - 6
        y = max(y, 14)

        # text background
        (tw, th), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(output, (x, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
        cv2.putText(output, class_name, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return output


def process_directory(input_dir: Path, output_dir: Path, alpha: float = 0.45):
    """Find all .json files in input_dir, render masks, save to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} annotation files in {input_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        shapes = data.get("shapes", [])
        if not shapes:
            continue

        # find the matching image
        img_name = data.get("imagePath", "")
        img_path = input_dir / img_name
        if not img_path.exists():
            # try same stem with common extensions
            for ext in [".jpg", ".png", ".jpeg", ".bmp"]:
                candidate = input_dir / (jf.stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            else:
                print(f"  skip {jf.name} -- image not found")
                continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  skip {jf.name} -- could not read image")
            continue

        result = render_masked_image(img, shapes, alpha=alpha)

        out_name = f"{jf.stem}-masked.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), result)

        n_shapes = len(shapes)
        classes = set(get_class_from_label(s["label"]) for s in shapes if s.get("points"))
        print(f"  {jf.stem}: {n_shapes} masks ({', '.join(sorted(classes))}) -> {out_name}")

    print("-" * 60)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render filled colored mask overlays from SAM JSON outputs")
    parser.add_argument("--input", required=True, help="Directory with .json + image files")
    parser.add_argument("--output", default=None, help="Output directory (default: <input>/masked)")
    parser.add_argument("--alpha", type=float, default=0.45, help="Mask transparency (0-1, default 0.45)")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir / "masked"

    process_directory(input_dir, output_dir, alpha=args.alpha)
