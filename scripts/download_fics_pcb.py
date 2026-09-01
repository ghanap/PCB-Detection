#!/usr/bin/env python3
"""
download_fics_pcb.py

Inspects, organizes, and splits the FICS-PCB dataset into board-level train/val/test sets.
Grouping by board ID prevents data leakage across splits.
"""

import os
import glob
import json
import shutil
import random
from pathlib import Path

DATASET_SRC = Path(r"C:\Users\ANAGHA\pcb-data\Data")
OUTPUT_DIR = Path(r"C:\Userdata\antiiii\dataset_split")

CLASS_NAMES = ['Cap1', 'Cap2', 'Cap3', 'Cap4', 'MOSFET', 'Mov', 'Resistor', 'Transformer']

def get_board_id(filename: str) -> str:
    """Extracts board identifier from filename (e.g. 'VID20210601143927-96...')."""
    base = os.path.basename(filename)
    if '-' in base:
        return base.split('-')[0]
    return base.split('.')[0]

def organize_and_split():
    print("Finding all images and annotations in source directory...")

    # Search for all image files
    all_images = list(DATASET_SRC.rglob("*.jpg")) + list(DATASET_SRC.rglob("*.png"))
    print(f"Found {len(all_images)} total images in source dataset.")

    # Group images by board ID
    board_map = {}
    for img_path in all_images:
        board_id = get_board_id(img_path.name)
        if board_id not in board_map:
            board_map[board_id] = []
        board_map[board_id].append(img_path)

    board_ids = sorted(list(board_map.keys()))
    print(f"Grouped dataset into {len(board_ids)} distinct PCB board models/scans:")
    for b_id in board_ids:
        print(f"  - Board '{b_id}': {len(board_map[b_id])} images")

    # Split board IDs into train (70%), val (15%), test (15%)
    random.seed(42)
    random.shuffle(board_ids)

    n_boards = len(board_ids)
    if n_boards >= 3:
        n_test = max(1, int(n_boards * 0.15))
        n_val = max(1, int(n_boards * 0.15))
        n_train = n_boards - n_val - n_test
    else:
        n_train, n_val, n_test = n_boards, 0, 0

    train_boards = board_ids[:n_train]
    val_boards = board_ids[n_train:n_train + n_val]
    test_boards = board_ids[n_train + n_val:]

    print("\nBoard-Level Split Allocation:")
    print(f"  Train ({len(train_boards)} boards): {train_boards}")
    print(f"  Val   ({len(val_boards)} boards): {val_boards}")
    print(f"  Test  ({len(test_boards)} boards): {test_boards}")

    # Create destination directory structure
    for split in ['train', 'val', 'test']:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "json_labels").mkdir(parents=True, exist_ok=True)

    # Copy files to split directories
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    split_boards = {'train': train_boards, 'val': val_boards, 'test': test_boards}

    for split, boards in split_boards.items():
        for b_id in boards:
            for img_path in board_map[b_id]:
                stem = img_path.stem
                parent = img_path.parent
                
                # Copy image
                dst_img = OUTPUT_DIR / split / "images" / img_path.name
                shutil.copy2(img_path, dst_img)

                # Look for matching JSON labelme file
                json_candidates = [
                    parent / f"{stem}.json",
                    img_path.with_suffix('.json')
                ]
                for j_cand in json_candidates:
                    if j_cand.exists():
                        shutil.copy2(j_cand, OUTPUT_DIR / split / "json_labels" / j_cand.name)
                        break

                # Look for matching YOLO txt label file
                txt_candidates = [
                    parent / f"{stem}.txt",
                    parent.parent / "labels" / f"{stem}.txt"
                ]
                for t_cand in txt_candidates:
                    if t_cand.exists():
                        shutil.copy2(t_cand, OUTPUT_DIR / split / "labels" / t_cand.name)
                        break

                split_counts[split] += 1

    print("\nDataset Organization Summary:")
    for split, count in split_counts.items():
        print(f"  {split.capitalize()} Set: {count} images")

    # Generate metadata YAML
    yaml_content = f"""path: {OUTPUT_DIR.resolve()}
train: train/images
val: val/images
test: test/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
    with open(OUTPUT_DIR / "fics_pcb.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\nSaved dataset configuration to {OUTPUT_DIR / 'fics_pcb.yaml'}")

if __name__ == "__main__":
    organize_and_split()
