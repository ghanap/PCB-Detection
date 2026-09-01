#!/usr/bin/env python3
"""
train.py

Fine-tunes YOLOv11-seg starting strictly from a COCO-pretrained checkpoint (yolo11n-seg.pt)
on the SAM-masked, patched PCB dataset. Incorporates on-the-fly data augmentations.
"""

import os
import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11-seg on patched, SAM-masked PCB dataset.")
    parser.add_argument("--data", type=str, default=r"C:\Userdata\antiiii\dataset_tiled\fics_pcb_tiled.yaml",
                        help="Path to dataset fics_pcb_tiled.yaml file")
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt",
                        help="COCO-pretrained YOLOv11-seg model checkpoint (yolo11n-seg.pt, yolo11s-seg.pt, etc.)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default 640)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--name", type=str, default="yolo11_fics_pcb_seg", help="Experiment name")
    parser.add_argument("--project", type=str, default=r"C:\Userdata\antiiii\runs", help="Output project directory")
    args = parser.parse_args()

    print(f"Initializing YOLOv11-seg from COCO-pretrained checkpoint: {args.model}...")
    model = YOLO(args.model)

    print("Configuring training parameters and on-the-fly augmentations...")
    # Training configuration with on-the-fly augmentations (mosaic, HSV jitter, scale/translate/blur)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        workers=0,  # Windows process stability
        device='cpu',  # CPU training
        # Data Augmentations
        mosaic=0.8,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        flipud=0.5,
        save=True,
        save_period=5,
        plots=True,
        verbose=True
    )

    print(f"\nTraining completed successfully! Model weights saved in: {Path(args.project) / args.name}")

if __name__ == "__main__":
    main()
