#!/usr/bin/env python3
"""Download PCB-Detection datasets from Hugging Face into their expected paths."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "cropped-pcbs": (
        "SanderGi/pcb-detection-cropped-pcbs",
        REPOSITORY_ROOT / "data" / "cropped_pcbs",
    ),
    "augmented-obb": (
        "SanderGi/pcb-detection-augmented-obb",
        REPOSITORY_ROOT / "data" / "augmented_obb",
    ),
    "augmented-seg": (
        "SanderGi/pcb-detection-augmented-seg",
        REPOSITORY_ROOT / "data" / "augmented_seg",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PCB-Detection datasets from Hugging Face."
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=sorted(DATASETS),
        help="Datasets to download. Defaults to all datasets.",
    )
    args = parser.parse_args()

    selected = args.datasets or list(DATASETS)
    for name in selected:
        repo_id, destination = DATASETS[name]
        destination.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {repo_id} to {destination}")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=destination,
        )


if __name__ == "__main__":
    main()
