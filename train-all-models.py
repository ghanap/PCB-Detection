#!/usr/bin/env python3
"""
train-all-models.py

Four pipelines for PCB component detection:

  1. YOLO BB         -- train YOLO for bounding box detection
  2. YOLO-seg (BB)   -- generate SAM masks using GT box prompts,
                        then train YOLO-seg on those pseudo-labels
  3. YOLO-seg (auto) -- generate SAM masks with no prompts (automatic),
                        then train YOLO-seg on those pseudo-labels
  4. YOLO -> SAM     -- use YOLO BB predictions to prompt SAM at inference

Steps can be run individually with --step or all together.

Usage:
  python train-all-models.py                          # run all 4
  python train-all-models.py --step 1                 # YOLO BB only
  python train-all-models.py --step 2                 # generate SAM+BB labels + train seg
  python train-all-models.py --step 3                 # generate SAM-auto labels + train seg
  python train-all-models.py --step 4                 # YOLO->SAM inference pipeline
  python train-all-models.py --generate-only          # only generate pseudo-labels, skip training
  python train-all-models.py --step 2 --limit 20      # test on 20 images
"""

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(r"c:\Userdata\antiiii")
DATASET    = BASE / "dataset_split"
DATA_YAML  = DATASET / "fics_pcb.yaml"
SAM_CKPT   = BASE / "sam_vit_b.pth"
YOLO_BASE_DET = "yolo11n.pt"
YOLO_BASE_SEG = "yolo11n-seg.pt"
OUT_ROOT   = BASE / "model-outputs"

CLASS_MAP = {0:'Cap1',1:'Cap2',2:'Cap3',3:'Cap4',
             4:'MOSFET',5:'Mov',6:'Resistor',7:'Transformer'}
NUM_CLASSES = 8
CLASS_NAMES = [CLASS_MAP[i] for i in range(NUM_CLASSES)]

SPLITS = ["train", "val", "test"]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_yolo_labels(label_path, img_w, img_h):
    """Read YOLO-format label file -> list of (class_id, [x1,y1,x2,y2])."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = max(0, int(round((xc - bw/2) * img_w)))
            y1 = max(0, int(round((yc - bh/2) * img_h)))
            x2 = min(img_w, int(round((xc + bw/2) * img_w)))
            y2 = min(img_h, int(round((yc + bh/2) * img_h)))
            if x2 > x1 + 4 and y2 > y1 + 4:
                boxes.append((cid, [x1, y1, x2, y2]))
    return boxes


def mask_to_yolo_seg(mask, class_id, img_w, img_h, min_area=12.0):
    """Convert binary mask to YOLO segmentation label line.
    Returns a string like '3 0.12 0.34 0.56 0.78 ...' or None if invalid."""
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area:
        return None
    approx = cv2.approxPolyDP(cnt, 0.005 * cv2.arcLength(cnt, True), True)
    if len(approx) < 3:
        return None
    # normalize to 0-1
    coords = []
    for p in approx:
        x_norm = round(float(p[0][0]) / img_w, 6)
        y_norm = round(float(p[0][1]) / img_h, 6)
        coords.extend([x_norm, y_norm])
    return f"{class_id} " + " ".join(str(c) for c in coords)


def box_to_yolo_seg(box, class_id, img_w, img_h):
    """Fallback: convert box [x1,y1,x2,y2] to 4-point polygon YOLO-seg line."""
    x1, y1, x2, y2 = box
    coords = [
        round(x1/img_w, 6), round(y1/img_h, 6),
        round(x2/img_w, 6), round(y1/img_h, 6),
        round(x2/img_w, 6), round(y2/img_h, 6),
        round(x1/img_w, 6), round(y2/img_h, 6),
    ]
    return f"{class_id} " + " ".join(str(c) for c in coords)


def compute_iou(mask_a, mask_b):
    inter = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(inter / union) if union > 0 else 0.0


def get_image_files(split_dir):
    """Get clean image files (skip SAM outputs, overlays, etc)."""
    files = []
    for f in sorted(split_dir.iterdir()):
        if f.suffix.lower() not in ('.jpg', '.png', '.jpeg'):
            continue
        if any(tag in f.stem for tag in ['_sam', '_mask', '_overlay', 'legend']):
            continue
        files.append(f)
    return files


def write_data_yaml(out_dir, dataset_name):
    """Write a data.yaml for a YOLO-seg dataset."""
    cfg = {
        "path": str(out_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": NUM_CLASSES,
        "names": CLASS_NAMES,
    }
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"  Wrote {yaml_path}")
    return yaml_path


# ===================================================================
# STEP 1: YOLO Bounding Box Detection
# ===================================================================

def step1_yolo_bb(epochs=50, batch=16, existing_weights=None):
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("STEP 1: YOLO BOUNDING BOX DETECTION")
    print("=" * 70)

    out_dir = OUT_ROOT / "step1-yolo-bb"
    out_dir.mkdir(parents=True, exist_ok=True)

    if existing_weights:
        weights = Path(existing_weights)
        print(f"Using existing weights: {weights}")
    else:
        model = YOLO(YOLO_BASE_DET)
        print(f"Training YOLOv11n-det | epochs={epochs} batch={batch}")
        model.train(
            data=str(DATA_YAML),
            epochs=epochs,
            imgsz=640,
            batch=batch,
            project=str(out_dir),
            name="train",
            exist_ok=True,
            save=True,
            plots=True,
            patience=15,
            hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
            degrees=5.0, translate=0.1, scale=0.3,
            fliplr=0.5, flipud=0.2, mosaic=0.8,
        )
        weights = out_dir / "train" / "weights" / "best.pt"

    # evaluate
    print("\nEvaluating on test split...")
    model = YOLO(str(weights))
    metrics = model.val(data=str(DATA_YAML), split="test",
                        project=str(out_dir), name="test-eval", exist_ok=True)

    summary = {
        "model": "YOLO-BB",
        "weights": str(weights),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50-95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  mAP@50:    {summary['mAP50']}")
    print(f"  mAP@50-95: {summary['mAP50-95']}")
    return weights, summary


# ===================================================================
# STEP 2: Generate SAM+BB pseudo-labels, then train YOLO-seg
# ===================================================================

def step2_generate_sam_bb_labels(limit=None):
    """Use SAM with GT bounding box prompts to create segmentation labels."""
    from segment_anything import sam_model_registry, SamPredictor

    print("\n" + "=" * 70)
    print("STEP 2a: GENERATE SAM PSEUDO-LABELS (WITH BB PROMPTS)")
    print("=" * 70)

    seg_dataset = OUT_ROOT / "dataset-sam-bb"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM ViT-B on {device}...")
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    total_masks = 0
    for split in SPLITS:
        img_dir = DATASET / split / "images"
        lbl_dir = DATASET / split / "labels"
        out_img_dir = seg_dataset / split / "images"
        out_lbl_dir = seg_dataset / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_files = get_image_files(img_dir)
        if limit:
            img_files = img_files[:limit]

        print(f"\n  {split}: processing {len(img_files)} images...")
        t0 = time.time()

        for i, img_p in enumerate(img_files, 1):
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            h, w = img.shape[:2]
            boxes = parse_yolo_labels(lbl_dir / f"{img_p.stem}.txt", w, h)
            if not boxes:
                continue

            predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            seg_lines = []

            for cid, box in boxes:
                masks, scores, _ = predictor.predict(
                    box=np.array(box)[None, :], multimask_output=False)
                line = mask_to_yolo_seg(masks[0], cid, w, h)
                if line is None:
                    line = box_to_yolo_seg(box, cid, w, h)
                seg_lines.append(line)

            # symlink or copy image, write label
            dst_img = out_img_dir / img_p.name
            if not dst_img.exists():
                shutil.copy2(img_p, dst_img)
            with open(out_lbl_dir / f"{img_p.stem}.txt", "w") as f:
                f.write("\n".join(seg_lines))
            total_masks += len(seg_lines)

            if i % 50 == 0 or i == len(img_files):
                print(f"    [{i}/{len(img_files)}] {time.time()-t0:.0f}s")

    yaml_path = write_data_yaml(seg_dataset, "sam-bb")
    print(f"\n  Total mask labels generated: {total_masks}")
    print(f"  Dataset saved to: {seg_dataset}")
    return seg_dataset, yaml_path


def step2_train_yolo_seg_bb(yaml_path, epochs=50, batch=16):
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("STEP 2b: TRAIN YOLO-SEG ON SAM+BB PSEUDO-LABELS")
    print("=" * 70)

    out_dir = OUT_ROOT / "step2-yolo-seg-bb"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(YOLO_BASE_SEG)
    print(f"Training YOLOv11n-seg | epochs={epochs} batch={batch}")
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        project=str(out_dir),
        name="train",
        exist_ok=True,
        save=True,
        plots=True,
        patience=15,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        degrees=5.0, translate=0.1, scale=0.3,
        fliplr=0.5, flipud=0.2, mosaic=0.8,
    )

    weights = out_dir / "train" / "weights" / "best.pt"

    print("\nEvaluating on test split...")
    model = YOLO(str(weights))
    metrics = model.val(data=str(yaml_path), split="test",
                        project=str(out_dir), name="test-eval", exist_ok=True)

    summary = {
        "model": "YOLO-seg (SAM+BB labels)",
        "weights": str(weights),
        "box_mAP50": round(float(metrics.box.map50), 4),
        "box_mAP50-95": round(float(metrics.box.map), 4),
        "mask_mAP50": round(float(metrics.seg.map50), 4) if hasattr(metrics, 'seg') else None,
        "mask_mAP50-95": round(float(metrics.seg.map), 4) if hasattr(metrics, 'seg') else None,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Box  mAP@50: {summary['box_mAP50']}")
    print(f"  Mask mAP@50: {summary['mask_mAP50']}")
    return weights, summary


# ===================================================================
# STEP 3: Generate SAM-auto pseudo-labels, then train YOLO-seg
# ===================================================================

def step3_generate_sam_auto_labels(limit=None):
    """Use SAM automatic mask generator, match masks to GT boxes."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    print("\n" + "=" * 70)
    print("STEP 3a: GENERATE SAM PSEUDO-LABELS (AUTOMATIC, NO PROMPTS)")
    print("=" * 70)

    seg_dataset = OUT_ROOT / "dataset-sam-auto"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM ViT-B on {device}...")
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT))
    sam.to(device=device)
    sam.eval()

    generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=100,
    )

    total_masks = 0
    total_missed = 0

    for split in SPLITS:
        img_dir = DATASET / split / "images"
        lbl_dir = DATASET / split / "labels"
        out_img_dir = seg_dataset / split / "images"
        out_lbl_dir = seg_dataset / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_files = get_image_files(img_dir)
        if limit:
            img_files = img_files[:limit]

        print(f"\n  {split}: processing {len(img_files)} images...")
        t0 = time.time()

        for i, img_p in enumerate(img_files, 1):
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            h, w = img.shape[:2]
            gt_boxes = parse_yolo_labels(lbl_dir / f"{img_p.stem}.txt", w, h)
            if not gt_boxes:
                continue

            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            auto_masks = generator.generate(rgb)

            seg_lines = []
            used_mask_indices = set()

            # for each GT box, find best matching auto mask
            for cid, gt_box in gt_boxes:
                gt_mask = np.zeros((h, w), dtype=np.uint8)
                gt_mask[gt_box[1]:gt_box[3], gt_box[0]:gt_box[2]] = 1

                best_iou = 0.0
                best_idx = -1
                for midx, m in enumerate(auto_masks):
                    if midx in used_mask_indices:
                        continue
                    iou = compute_iou(m["segmentation"], gt_mask)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = midx

                if best_iou > 0.25 and best_idx >= 0:
                    used_mask_indices.add(best_idx)
                    line = mask_to_yolo_seg(
                        auto_masks[best_idx]["segmentation"], cid, w, h)
                    if line:
                        seg_lines.append(line)
                    else:
                        seg_lines.append(box_to_yolo_seg(gt_box, cid, w, h))
                        total_missed += 1
                else:
                    # no good auto mask found, fall back to box
                    seg_lines.append(box_to_yolo_seg(gt_box, cid, w, h))
                    total_missed += 1

            dst_img = out_img_dir / img_p.name
            if not dst_img.exists():
                shutil.copy2(img_p, dst_img)
            with open(out_lbl_dir / f"{img_p.stem}.txt", "w") as f:
                f.write("\n".join(seg_lines))
            total_masks += len(seg_lines)

            if i % 10 == 0 or i == len(img_files):
                print(f"    [{i}/{len(img_files)}] {time.time()-t0:.0f}s | "
                      f"auto masks/img: {len(auto_masks)}")

    yaml_path = write_data_yaml(seg_dataset, "sam-auto")
    print(f"\n  Total labels generated: {total_masks}")
    print(f"  Fell back to box (no good auto mask): {total_missed}")
    print(f"  Dataset saved to: {seg_dataset}")
    return seg_dataset, yaml_path


def step3_train_yolo_seg_auto(yaml_path, epochs=50, batch=16):
    from ultralytics import YOLO

    print("\n" + "=" * 70)
    print("STEP 3b: TRAIN YOLO-SEG ON SAM-AUTO PSEUDO-LABELS")
    print("=" * 70)

    out_dir = OUT_ROOT / "step3-yolo-seg-auto"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(YOLO_BASE_SEG)
    print(f"Training YOLOv11n-seg | epochs={epochs} batch={batch}")
    model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        project=str(out_dir),
        name="train",
        exist_ok=True,
        save=True,
        plots=True,
        patience=15,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        degrees=5.0, translate=0.1, scale=0.3,
        fliplr=0.5, flipud=0.2, mosaic=0.8,
    )

    weights = out_dir / "train" / "weights" / "best.pt"

    print("\nEvaluating on test split...")
    model = YOLO(str(weights))
    metrics = model.val(data=str(yaml_path), split="test",
                        project=str(out_dir), name="test-eval", exist_ok=True)

    summary = {
        "model": "YOLO-seg (SAM-auto labels)",
        "weights": str(weights),
        "box_mAP50": round(float(metrics.box.map50), 4),
        "box_mAP50-95": round(float(metrics.box.map), 4),
        "mask_mAP50": round(float(metrics.seg.map50), 4) if hasattr(metrics, 'seg') else None,
        "mask_mAP50-95": round(float(metrics.seg.map), 4) if hasattr(metrics, 'seg') else None,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Box  mAP@50: {summary['box_mAP50']}")
    print(f"  Mask mAP@50: {summary['mask_mAP50']}")
    return weights, summary


# ===================================================================
# STEP 4: YOLO -> SAM inference pipeline
# ===================================================================

def step4_yolo_then_sam(yolo_weights=None):
    """Run YOLO BB to detect, feed predictions into SAM for masks."""
    from ultralytics import YOLO
    from segment_anything import sam_model_registry, SamPredictor

    print("\n" + "=" * 70)
    print("STEP 4: YOLO -> SAM INFERENCE PIPELINE")
    print("=" * 70)

    out_dir = OUT_ROOT / "step4-yolo-sam-pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    if yolo_weights is None:
        # try step1 output first, then existing weights
        candidate = OUT_ROOT / "step1-yolo-bb" / "train" / "weights" / "best.pt"
        if candidate.exists():
            yolo_weights = candidate
        else:
            yolo_weights = BASE / "runs" / "yolo11_fics_pcb_seg" / "weights" / "best.pt"

    print(f"YOLO weights: {yolo_weights}")
    yolo = YOLO(str(yolo_weights))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM ViT-B on {device}...")
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CKPT))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    test_imgs = DATASET / "test" / "images"
    test_lbls = DATASET / "test" / "labels"
    img_files = get_image_files(test_imgs)

    all_records = []
    all_ious = []
    t0 = time.time()

    for i, img_p in enumerate(img_files, 1):
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        h, w = img.shape[:2]

        # YOLO prediction
        results = yolo.predict(str(img_p), conf=0.25, verbose=False)
        if not results or len(results[0].boxes) == 0:
            continue

        predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        det_boxes = results[0].boxes

        shapes = []
        for j in range(len(det_boxes)):
            box = det_boxes.xyxy[j].cpu().numpy().astype(int).tolist()
            cid = int(det_boxes.cls[j].item())
            det_conf = float(det_boxes.conf[j].item())

            masks, scores, _ = predictor.predict(
                box=np.array(box)[None, :], multimask_output=False)
            sam_conf = float(scores[0])

            line = mask_to_yolo_seg(masks[0], cid, w, h)
            # compute IoU against GT if available
            gt_boxes = parse_yolo_labels(test_lbls / f"{img_p.stem}.txt", w, h)
            best_gt_iou = 0.0
            for _, gt_box in gt_boxes:
                gt_mask = np.zeros((h, w), dtype=np.uint8)
                gt_mask[gt_box[1]:gt_box[3], gt_box[0]:gt_box[2]] = 1
                iou = compute_iou(masks[0], gt_mask)
                best_gt_iou = max(best_gt_iou, iou)

            all_ious.append(best_gt_iou)
            class_name = CLASS_MAP.get(cid, f"Class_{cid}")
            all_records.append({
                "image": img_p.name,
                "class": class_name,
                "yolo_conf": round(det_conf, 3),
                "sam_conf": round(sam_conf, 3),
                "best_gt_iou": round(best_gt_iou, 3),
            })

        if i % 50 == 0 or i == len(img_files):
            print(f"  [{i}/{len(img_files)}] {time.time()-t0:.0f}s | "
                  f"mean IoU: {np.mean(all_ious):.3f}")

    df = pd.DataFrame(all_records)
    df.to_csv(out_dir / "pipeline-results.csv", index=False)

    summary = {
        "model": "YOLO->SAM pipeline",
        "yolo_weights": str(yolo_weights),
        "images": len(img_files),
        "detections": len(all_records),
        "mean_yolo_conf": round(float(df["yolo_conf"].mean()), 3),
        "mean_sam_conf": round(float(df["sam_conf"].mean()), 3),
        "mean_iou": round(float(np.mean(all_ious)), 3) if all_ious else 0,
        "median_iou": round(float(np.median(all_ious)), 3) if all_ious else 0,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Detections: {summary['detections']}")
    print(f"  Mean YOLO conf: {summary['mean_yolo_conf']}")
    print(f"  Mean SAM conf:  {summary['mean_sam_conf']}")
    print(f"  Mean IoU vs GT: {summary['mean_iou']}")
    return summary


# ===================================================================
# Comparison table
# ===================================================================

def build_comparison():
    """Load results.json from each step and print comparison."""
    print("\n" + "=" * 70)
    print("BENCHMARK COMPARISON")
    print("=" * 70)

    rows = []
    for name, path in [
        ("YOLO BB",              OUT_ROOT / "step1-yolo-bb" / "results.json"),
        ("YOLO-seg (SAM+BB)",    OUT_ROOT / "step2-yolo-seg-bb" / "results.json"),
        ("YOLO-seg (SAM-auto)",  OUT_ROOT / "step3-yolo-seg-auto" / "results.json"),
        ("YOLO->SAM pipeline",   OUT_ROOT / "step4-yolo-sam-pipeline" / "results.json"),
    ]:
        if path.exists():
            with open(path) as f:
                rows.append({"Model": name, **json.load(f)})
        else:
            rows.append({"Model": name, "status": "not run"})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT_ROOT / "benchmark-comparison.csv", index=False)
    print(f"\nSaved to {OUT_ROOT / 'benchmark-comparison.csv'}")


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=[1,2,3,4], default=None,
                        help="Run only this step (1-4)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit images for label generation (testing)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate pseudo-labels, skip YOLO training")
    parser.add_argument("--yolo-weights", default=None,
                        help="Existing YOLO-BB weights (skip step 1 training)")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Dataset: {DATASET}  ({sum(len(get_image_files(DATASET/s/'images')) for s in SPLITS)} images)")
    print(f"Output:  {OUT_ROOT}")

    yolo_weights = None

    # Step 1
    if args.step is None or args.step == 1:
        yolo_weights, _ = step1_yolo_bb(
            epochs=args.epochs, batch=args.batch,
            existing_weights=args.yolo_weights)

    # Step 2
    if args.step is None or args.step == 2:
        _, yaml2 = step2_generate_sam_bb_labels(limit=args.limit)
        if not args.generate_only:
            step2_train_yolo_seg_bb(yaml2, epochs=args.epochs, batch=args.batch)

    # Step 3
    if args.step is None or args.step == 3:
        _, yaml3 = step3_generate_sam_auto_labels(limit=args.limit)
        if not args.generate_only:
            step3_train_yolo_seg_auto(yaml3, epochs=args.epochs, batch=args.batch)

    # Step 4
    if args.step is None or args.step == 4:
        step4_yolo_then_sam(yolo_weights=yolo_weights or args.yolo_weights)

    # Comparison
    build_comparison()
    print("\nDone.")
