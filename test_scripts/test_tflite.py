"""Run offline inference on a retina image using the DRISHTI TFLite model."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import tensorflow as tf

from preprocessing.preprocess import preprocess_image
from project_paths import SUPPORTED_IMAGE_SUFFIXES, resolve_data_root

CLASS_NAMES = ['glaucoma', 'normal']


def resolve_image_path(image_path: str | None) -> Path:
    """Resolve a requested image or fall back to the first validation/training image."""
    if image_path:
        candidate = Path(image_path)
        if candidate.is_dir():
            matches = sorted(path for path in candidate.rglob('*') if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)
            if not matches:
                raise FileNotFoundError(f'No supported images found under directory: {candidate}')
            return matches[0]
        return candidate

    data_root = resolve_data_root('data')
    for split in ['val', 'train']:
        matches = sorted(path for path in (data_root / split).rglob('*') if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)
        if matches:
            return matches[0]

    raise FileNotFoundError(
        'No input image was provided and no sample image was found under data/val or data/train. '
        'Pass --image <path-to-image>.'
    )


def run_inference(model_path: str, image_path: str | None = None) -> tuple[Path, str, float, np.ndarray]:
    """Load a TFLite model, run inference, and return the label and confidence."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    resolved_image = resolve_image_path(image_path)
    image = cv2.imread(str(resolved_image))
    if image is None:
        raise FileNotFoundError(f'Image not found: {resolved_image}')

    processed = preprocess_image(image)
    input_tensor = np.expand_dims(processed, axis=0).astype(input_details['dtype'])

    interpreter.set_tensor(input_details['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details['index'])[0]

    predicted_index = int(np.argmax(output))
    confidence = float(output[predicted_index])
    return resolved_image, CLASS_NAMES[predicted_index], confidence, output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test DRISHTI TFLite inference.')
    parser.add_argument('--model', default='tflite_model/drishti_model.tflite', help='Path to .tflite file')
    parser.add_argument('--image', help='Path to one retina image, or a directory to auto-pick the first image')
    args = parser.parse_args()

    image_path, label, confidence, raw_output = run_inference(args.model, args.image)
    print(f'Image: {image_path}')
    print(f'Prediction: {label}')
    print(f'Confidence: {confidence:.4f}')
    print(f'Raw probabilities: {raw_output}')
