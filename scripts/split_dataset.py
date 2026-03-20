"""Create or repair stratified train/val splits for DRISHTI datasets."""
from __future__ import annotations

import argparse
from hashlib import sha256
import random
import shutil
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project_paths import SUPPORTED_IMAGE_SUFFIXES, ensure_dataset_layout


CLASS_NAMES = ('glaucoma', 'normal')


def file_digest(path: Path) -> str:
    digest = sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_layout(root: Path) -> str:
    if all((root / class_name).exists() for class_name in CLASS_NAMES):
        return 'flat'
    if (root / 'train').exists() or (root / 'val').exists():
        return 'split'
    raise ValueError(
        'Input dataset must either contain glaucoma/ and normal/ folders, or train/ and val/ split folders.'
    )


def iter_class_images(root: Path, class_name: str) -> Iterable[Path]:
    layout = source_layout(root)
    if layout == 'flat':
        search_roots = [root / class_name]
    else:
        search_roots = [root / 'train' / class_name, root / 'val' / class_name]

    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob('*'):
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
                yield path


def list_unique_images(root: Path, class_name: str) -> list[Path]:
    unique: Dict[str, Path] = {}
    for path in iter_class_images(root, class_name):
        unique.setdefault(file_digest(path), path)
    return sorted(unique.values())


def stratified_split(items: list[Path], val_ratio: float) -> tuple[list[Path], list[Path]]:
    if not items:
        return [], []
    val_count = int(round(len(items) * val_ratio))
    if len(items) > 1:
        val_count = max(1, min(len(items) - 1, val_count))
    else:
        val_count = 0
    return items[val_count:], items[:val_count]


def copy_split(items: Iterable[Path], class_name: str, destination_root: Path, split: str) -> int:
    copied = 0
    for index, source_path in enumerate(items, start=1):
        destination = destination_root / split / class_name / f'{class_name}_{split}_{index}{source_path.suffix.lower()}'
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        copied += 1
    return copied


def build_split(source_root: Path, destination_root: Path, val_ratio: float, seed: int) -> Dict[str, Dict[str, int]]:
    rng = random.Random(seed)
    ensure_dataset_layout(destination_root)
    summary: Dict[str, Dict[str, int]] = {'train': {}, 'val': {}}

    for class_name in CLASS_NAMES:
        items = list_unique_images(source_root, class_name)
        if not items:
            raise ValueError(f'No images found for class: {class_name} under {source_root}')
        rng.shuffle(items)
        train_items, val_items = stratified_split(items, val_ratio)
        summary['train'][class_name] = copy_split(train_items, class_name, destination_root, 'train')
        summary['val'][class_name] = copy_split(val_items, class_name, destination_root, 'val')
    return summary


def rebuild_in_place(dataset_root: Path, val_ratio: float, seed: int) -> Dict[str, Dict[str, int]]:
    with TemporaryDirectory(dir=dataset_root.parent, prefix=f'{dataset_root.name}_repair_') as tmpdir:
        temp_root = Path(tmpdir) / dataset_root.name
        summary = build_split(dataset_root, temp_root, val_ratio, seed)

        for split in ('train', 'val'):
            split_dir = dataset_root / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
        shutil.move(str(temp_root / 'train'), str(dataset_root / 'train'))
        shutil.move(str(temp_root / 'val'), str(dataset_root / 'val'))
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Create or repair train/val splits for DRISHTI datasets.')
    parser.add_argument('--input', required=True, help='Source dataset root (flat class folders or existing train/val layout)')
    parser.add_argument('--output', default='data', help='Destination dataset root when not using --in-place')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Fraction of each class reserved for validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for deterministic shuffling')
    parser.add_argument('--in-place', action='store_true', help='Repair an existing dataset root in place')
    args = parser.parse_args()

    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError('--val-ratio must be in the range [0.0, 1.0).')

    source_root = Path(args.input)
    if args.in_place:
        summary = rebuild_in_place(source_root, args.val_ratio, args.seed)
    else:
        destination_root = Path(args.output)
        if destination_root.exists() and any(destination_root.iterdir()):
            raise ValueError(f'Output directory must be empty or absent: {destination_root}')
        summary = build_split(source_root, destination_root, args.val_ratio, args.seed)
    print(summary)


if __name__ == '__main__':
    main()
