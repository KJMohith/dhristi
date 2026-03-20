"""Image quality assessment module for retina acquisition."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class QualityResult:
    is_good: bool
    blur_score: float
    brightness_score: float
    center_alignment_score: float
    reasons: Tuple[str, ...]


def detect_blur(image: np.ndarray, threshold: float = 80.0) -> Tuple[bool, float]:
    """Use Laplacian variance to measure blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score >= threshold, float(score)


def detect_brightness(image: np.ndarray, min_v: float = 45.0, max_v: float = 220.0) -> Tuple[bool, float]:
    """Estimate image brightness from the HSV V channel."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    score = float(np.mean(v_channel))
    return min_v <= score <= max_v, score


def center_alignment_score(image: np.ndarray) -> Tuple[bool, float]:
    """Approximate alignment by checking whether the brightest disc-like region is near center."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)

    if moments['m00'] == 0:
        return False, 0.0

    cx = moments['m10'] / moments['m00']
    cy = moments['m01'] / moments['m00']
    image_center = np.array([image.shape[1] / 2.0, image.shape[0] / 2.0])
    disc_center = np.array([cx, cy])
    distance = np.linalg.norm(disc_center - image_center)
    max_distance = np.linalg.norm(image_center)
    score = 1.0 - min(distance / max_distance, 1.0)
    return score >= 0.45, float(score)


def evaluate_quality(image: np.ndarray) -> QualityResult:
    """Combine blur, brightness, and center alignment into a boolean quality decision."""
    blur_ok, blur_score = detect_blur(image)
    brightness_ok, brightness_score = detect_brightness(image)
    center_ok, center_score = center_alignment_score(image)

    reasons = []
    if not blur_ok:
        reasons.append('Image too blurry')
    if not brightness_ok:
        reasons.append('Brightness out of range')
    if not center_ok:
        reasons.append('Optic disc or retina not centered well')

    return QualityResult(
        is_good=blur_ok and brightness_ok and center_ok,
        blur_score=blur_score,
        brightness_score=brightness_score,
        center_alignment_score=center_score,
        reasons=tuple(reasons),
    )


def evaluate_image_path(image_path: str) -> Dict[str, object]:
    """Convenience wrapper for CLI and app-side reuse."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Cannot load image: {image_path}')
    result = evaluate_quality(image)
    return {
        'is_good': result.is_good,
        'blur_score': result.blur_score,
        'brightness_score': result.brightness_score,
        'center_alignment_score': result.center_alignment_score,
        'reasons': list(result.reasons),
    }


if __name__ == '__main__':
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Run fundus image quality checks.')
    parser.add_argument('--image', required=True, help='Path to retina image')
    args = parser.parse_args()

    print(json.dumps(evaluate_image_path(args.image), indent=2))
