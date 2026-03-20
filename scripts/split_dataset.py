"""Create stratified train/val splits from a class-folder dataset."""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
import sys
from typing import Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import SUPPORTED_IMAGE_SUFFIXES, ensure_dataset_layout


CLASS_NAMES = ('glaucoma', 'normal')


def list_images(root: Path, class_name: str) -> list[Path]:
    class_dir = root / class_name
    if not class_dir.exists():
        return []
    return sorted(
        [path for path in class_dir.rglob('*') if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES]
    )


def stratified_split(items: list[Path], val_ratio: float) -> tuple[list[Path], list[Path]]:
    if not items:
        return [], []
    val_count = int(round(len(items) * val_ratio))
    if len(items) > 1:
        val_count = max(1, min(len(items) - 1, val_count))
    else:
        val_count = 0
    return items[val_count:], items[:val_count]


def copy_split(items: Iterable[Path], source_root: Path, destination_root: Path, split: str, class_name: str) -> int:
    copied = 0
    for source_path in items:
        relative = source_path.relative_to(source_root / class_name)
        destination = destination_root / split / class_name / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied += 1
    return copied


def build_split(source_root: Path, destination_root: Path, val_ratio: float, seed: int) -> Dict[str, Dict[str, int]]:
    rng = random.Random(seed)
    ensure_dataset_layout(destination_root)
    summary: Dict[str, Dict[str, int]] = {'train': {}, 'val': {}}

    for class_name in CLASS_NAMES:
        items = list_images(source_root, class_name)
        if not items:
            raise ValueError(f'No images found for class: {class_name} under {source_root / class_name}')
        rng.shuffle(items)
        train_items, val_items = stratified_split(items, val_ratio)
        summary['train'][class_name] = copy_split(train_items, source_root, destination_root, 'train', class_name)
        summary['val'][class_name] = copy_split(val_items, source_root, destination_root, 'val', class_name)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Create train/val splits from one folder per class.')
    parser.add_argument('--input', required=True, help='Source dataset root with glaucoma/ and normal/ folders')
    parser.add_argument('--output', default='data', help='Destination dataset root with train/ and val/ folders')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Fraction of each class reserved for validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic shuffling')
    args = parser.parse_args()

    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError('--val-ratio must be in the range [0.0, 1.0).')

    source_root = Path(args.input)
    destination_root = Path(args.output)
    if destination_root.exists() and any(destination_root.iterdir()):
        raise ValueError(f'Output directory must be empty or absent: {destination_root}')

    summary = build_split(source_root, destination_root, args.val_ratio, args.seed)
    print(summary)


if __name__ == '__main__':
    main()
