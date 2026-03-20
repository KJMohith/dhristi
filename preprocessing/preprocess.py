"""Fundus image preprocessing utilities for the DRISHTI MVP.

This module prepares retinal images for both training and on-device inference.
It includes retina cropping, smoothing, contrast enhancement, resizing,
and normalization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np

TARGET_SIZE: Tuple[int, int] = (224, 224)


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image from disk in BGR format."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    return image


def crop_retina_region(image: np.ndarray) -> np.ndarray:
    """Attempt to isolate the circular retina region using thresholding.

    Falls back to the original image if a reasonable retina contour is not found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 0.15 * image.shape[0] * image.shape[1]:
        return image

    (x, y), radius = cv2.minEnclosingCircle(largest)
    center = (int(x), int(y))
    radius = int(radius)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    cropped = cv2.bitwise_and(image, image, mask=mask)

    x1 = max(center[0] - radius, 0)
    y1 = max(center[1] - radius, 0)
    x2 = min(center[0] + radius, image.shape[1])
    y2 = min(center[1] + radius, image.shape[0])
    roi = cropped[y1:y2, x1:x2]
    return roi if roi.size else image


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """Enhance local contrast using CLAHE in LAB color space."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    enhanced = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Apply the full preprocessing pipeline to a fundus image.

    Steps:
    1. Crop retina circle when detected.
    2. Reduce high-frequency noise via Gaussian blur.
    3. Improve local contrast with CLAHE.
    4. Resize to 224x224.
    5. Normalize to [0, 1].
    """
    image = crop_retina_region(image)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = apply_clahe(image)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    return image


def save_preprocessed_image(input_path: str | Path, output_path: str | Path) -> None:
    """Load, preprocess, and save a visualization-friendly uint8 image."""
    processed = preprocess_image(load_image(input_path))
    output = (processed * 255).clip(0, 255).astype(np.uint8)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), output)


def preprocess_dataset_split(input_dir: str | Path, output_dir: str | Path) -> int:
    """Preprocess all supported images under a train/val split folder."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    image_paths: Iterable[Path] = [
        path for path in input_dir.rglob('*') if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}
    ]

    processed_count = 0
    for image_path in image_paths:
        relative = image_path.relative_to(input_dir)
        destination = output_dir / relative
        save_preprocessed_image(image_path, destination)
        processed_count += 1
    return processed_count


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess retinal fundus images.')
    parser.add_argument('--input', required=True, help='Input dataset split or image path')
    parser.add_argument('--output', required=True, help='Output dataset split or image path')
    parser.add_argument('--single', action='store_true', help='Treat input as one image instead of a folder')
    args = parser.parse_args()

    if args.single:
        save_preprocessed_image(args.input, args.output)
        print(f'Saved preprocessed image to {args.output}')
    else:
        count = preprocess_dataset_split(args.input, args.output)
        print(f'Processed {count} images into {args.output}')
