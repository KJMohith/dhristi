"""Dataset loading helpers for retinal fundus classification."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from preprocessing.preprocess import preprocess_image
from project_paths import SUPPORTED_IMAGE_SUFFIXES, resolve_data_root

CLASS_NAMES = ['glaucoma', 'normal']
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}


def load_dataset_split(split_dir: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a dataset split with folder names matching `CLASS_NAMES`."""
    split_path = Path(split_dir)
    images: List[np.ndarray] = []
    labels: List[int] = []

    for class_name in CLASS_NAMES:
        class_dir = split_path / class_name
        if not class_dir.exists():
            continue
        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            images.append(preprocess_image(image))
            labels.append(CLASS_TO_INDEX[class_name])

    if not images:
        raise ValueError(f'No images found in split: {split_dir}')

    return np.stack(images), np.array(labels, dtype=np.int32)


def summarize_dataset(dataset_root: str | Path) -> Dict[str, Dict[str, int]]:
    """Return a simple class distribution summary for train/val folders."""
    root = resolve_data_root(dataset_root)
    summary: Dict[str, Dict[str, int]] = {}
    for split in ['train', 'val']:
        split_dir = root / split
        summary[split] = {}
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            count = len([p for p in class_dir.glob('*') if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES])
            summary[split][class_name] = count
    return summary


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Summarize the retinal image dataset layout.')
    parser.add_argument('--dataset_root', default='data', help='Root dataset folder')
    args = parser.parse_args()

    print(json.dumps(summarize_dataset(args.dataset_root), indent=2))
