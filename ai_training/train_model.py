"""Train a MobileNetV3Small-based retina classifier for DRISHTI."""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras import layers

from preprocessing.preprocess import preprocess_image
from project_paths import SUPPORTED_IMAGE_SUFFIXES, resolve_data_root

IMAGE_SIZE: Tuple[int, int] = (224, 224)
BATCH_SIZE = 16
CLASS_NAMES = ['glaucoma', 'normal']
AUGMENTATION = keras.Sequential(
    [
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ],
    name='augmentation',
)


def _preprocess_batch(images: tf.Tensor, labels: tf.Tensor, training: bool = False) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply the shared fundus preprocessing pipeline to one dataset batch."""

    def _numpy_preprocess(batch):
        processed = [preprocess_image(image, target_size=IMAGE_SIZE) for image in batch]
        return tf.convert_to_tensor(processed, dtype=tf.float32).numpy()

    images = tf.numpy_function(_numpy_preprocess, [images], tf.float32)
    images.set_shape((None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    if training:
        images = AUGMENTATION(images, training=True)

    return images, labels


def build_datasets(dataset_root: str):
    """Create augmented training and deterministic validation datasets."""
    data_root = resolve_data_root(dataset_root)
    train_ds = keras.utils.image_dataset_from_directory(
        data_root / 'train',
        labels='inferred',
        label_mode='int',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        data_root / 'val',
        labels='inferred',
        label_mode='int',
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda x, y: _preprocess_batch(x, y, training=True), num_parallel_calls=autotune)
    val_ds = val_ds.map(lambda x, y: _preprocess_batch(x, y, training=False), num_parallel_calls=autotune)
    return train_ds.prefetch(autotune), val_ds.prefetch(autotune)


def compute_class_weight_map(data_root: Path) -> dict[int, float]:
    """Compute inverse-frequency class weights from the training split."""
    train_root = data_root / 'train'
    counts = {
        index: len([path for path in (train_root / class_name).glob('*') if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES])
        for index, class_name in enumerate(CLASS_NAMES)
    }
    total = sum(counts.values())
    if total == 0:
        raise ValueError(f'No training images found in {train_root}')
    num_classes = len(CLASS_NAMES)
    weights = {index: total / (num_classes * count) for index, count in counts.items() if count > 0}
    print(f'Using class weights: {weights}')
    return weights


def build_model(num_classes: int = 2) -> keras.Model:
    """Build a transfer learning model with MobileNetV3Small."""
    base_model = keras.applications.MobileNetV3Small(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights='imagenet',
    )
    base_model.trainable = False

    inputs = keras.Input(shape=IMAGE_SIZE + (3,))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')],
    )
    return model


def train(dataset_root: str, output_dir: str = 'ai_training/output', epochs: int = 10) -> Path:
    """Train the model and save the best Keras checkpoint."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data_root = resolve_data_root(dataset_root)
    train_ds, val_ds = build_datasets(str(data_root))
    class_weight = compute_class_weight_map(data_root)
    model = build_model(num_classes=len(CLASS_NAMES))

    best_model_path = output_path / 'best_model.keras'
    final_model_path = output_path / 'final_model.keras'

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor='val_accuracy',
            save_best_only=True,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )
    model.save(final_model_path)

    y_true = tf.concat([labels for _, labels in val_ds], axis=0).numpy().astype(int)
    y_pred = model.predict(val_ds, verbose=0).argmax(axis=1)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0)
    report_path = output_path / 'classification_report.txt'
    report_path.write_text(report)
    print(report)
    print(f'Model saved to {final_model_path}')
    print(f'Best model checkpoint: {best_model_path}')
    print(f'Classification report: {report_path}')
    return best_model_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train the DRISHTI retina classifier.')
    parser.add_argument('--dataset_root', default='data', help='Path to dataset root')
    parser.add_argument('--output_dir', default='ai_training/output', help='Where to save trained models')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    args = parser.parse_args()

    train(args.dataset_root, args.output_dir, args.epochs)
