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

CLASS_NAMES = ['glaucoma', 'normal']


def run_inference(model_path: str, image_path: str) -> tuple[str, float, np.ndarray]:
    """Load a TFLite model, run inference, and return the label and confidence."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f'Image not found: {image_path}')

    processed = preprocess_image(image)
    input_tensor = np.expand_dims(processed, axis=0).astype(input_details['dtype'])

    interpreter.set_tensor(input_details['index'], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details['index'])[0]

    predicted_index = int(np.argmax(output))
    confidence = float(output[predicted_index])
    return CLASS_NAMES[predicted_index], confidence, output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test DRISHTI TFLite inference.')
    parser.add_argument('--model', default='tflite_model/drishti_model.tflite', help='Path to .tflite file')
    parser.add_argument('--image', required=True, help='Path to retina image')
    args = parser.parse_args()

    label, confidence, raw_output = run_inference(args.model, args.image)
    print(f'Prediction: {label}')
    print(f'Confidence: {confidence:.4f}')
    print(f'Raw probabilities: {raw_output}')
