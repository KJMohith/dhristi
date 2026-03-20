"""Shared project path helpers for DRISHTI."""
from __future__ import annotations

from pathlib import Path

DEFAULT_DATA_DIRNAME = 'data'
LEGACY_DATA_DIRNAME = 'dataset'
SUPPORTED_IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg'}


def repo_root() -> Path:
    """Return the repository root directory."""
    return Path(__file__).resolve().parent


def resolve_data_root(requested: str | Path | None = None) -> Path:
    """Resolve the active dataset root, preferring `data/` over `dataset/`."""
    if requested is not None:
        candidate = Path(requested)
        if candidate.exists() or candidate.name not in {DEFAULT_DATA_DIRNAME, LEGACY_DATA_DIRNAME}:
            return candidate

    root = repo_root()
    preferred = root / DEFAULT_DATA_DIRNAME
    legacy = root / LEGACY_DATA_DIRNAME
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy
    return preferred


def ensure_dataset_layout(root: str | Path | None = None) -> Path:
    """Create the expected train/val class layout and return the root path."""
    data_root = resolve_data_root(root)
    for split in ['train', 'val']:
        for class_name in ['glaucoma', 'normal']:
            (data_root / split / class_name).mkdir(parents=True, exist_ok=True)
    return data_root
