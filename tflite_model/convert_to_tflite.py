"""Convert a trained Keras DRISHTI model into a quantized TFLite model."""
from __future__ import annotations

from pathlib import Path

import tensorflow as tf


def convert_model(model_path: str, output_path: str = 'tflite_model/drishti_model.tflite') -> Path:
    """Apply post-training quantization and save the .tflite artifact."""
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(tflite_model)

    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f'Saved TFLite model to {destination} ({size_mb:.2f} MB)')
    if size_mb > 5.0:
        print('Warning: model is larger than the <5MB target. Consider pruning or full int8 quantization.')
    return destination


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert Keras model to TFLite.')
    parser.add_argument('--model', default='ai_training/output/best_model.keras', help='Path to trained Keras model')
    parser.add_argument('--output', default='tflite_model/drishti_model.tflite', help='Output .tflite path')
    args = parser.parse_args()

    convert_model(args.model, args.output)
