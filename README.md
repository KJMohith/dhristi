# DRISHTI MVP

DRISHTI is an offline smartphone-based retinal fundus screening MVP designed for low-cost Android deployment. This repository includes Python tools for preprocessing, data quality checks, model training, TFLite conversion, and a Flutter app that performs on-device inference without cloud connectivity.

## What changed for run/deploy readiness

- `data/` is now the default dataset root.
- `scripts/bootstrap_project.py` can create `.venv/`, install Python dependencies, and generate a small synthetic dataset for smoke tests.
- Generated images and model artifacts are git-ignored so pull requests stay text-only and avoid binary file upload issues.
- `.gitattributes` marks common model/image artifacts as binary for safer tooling behavior.

## Final Project Structure

```text
DRISHTI/
├── ai_training/
│   └── train_model.py
├── data/
│   ├── README.md
│   ├── train/
│   │   ├── glaucoma/
│   │   └── normal/
│   └── val/
│       ├── glaucoma/
│       └── normal/
├── dataset/
│   └── ... legacy fallback layout ...
├── flutter_app/
│   ├── assets/models/
│   │   └── .gitkeep
│   ├── lib/
│   │   ├── screens/
│   │   ├── services/
│   │   └── widgets/
│   └── pubspec.yaml
├── preprocessing/
│   ├── dataset_loader.py
│   └── preprocess.py
├── quality_check/
│   └── quality_checker.py
├── scripts/
│   └── bootstrap_project.py
├── test_scripts/
│   └── test_tflite.py
├── tflite_model/
│   └── convert_to_tflite.py
├── project_paths.py
├── .gitattributes
├── .gitignore
├── README.md
└── requirements.txt
```

## Quick start

### 1. Bootstrap everything needed to run locally

```bash
python scripts/bootstrap_project.py
```

That command:
- creates `data/train/...` and `data/val/...`
- generates a small synthetic sample dataset for smoke testing
- creates `.venv/`
- installs dependencies from `requirements.txt`

If you already have dependencies installed, skip package installation:

```bash
python scripts/bootstrap_project.py --skip-install
```

### 2. Check the dataset summary

```bash
python preprocessing/dataset_loader.py --dataset_root data
```

The trainer now refuses to run if the same image appears in both `train/` and `val/`, so fix duplicated splits before training.

If you only have one folder per class and need a proper train/validation split, create one first:

```bash
python scripts/split_dataset.py --input raw_dataset --output data --val-ratio 0.2
```

If you accidentally copied the same images into both `train/` and `val/`, repair the existing dataset in place:

```bash
python scripts/split_dataset.py --input data --in-place --val-ratio 0.2
```

This deduplicates the source images by content and rebuilds a clean split.

### 3. Optionally preprocess the dataset

```bash
python preprocessing/preprocess.py --input data/train --output processed/train
python preprocessing/preprocess.py --input data/val --output processed/val
```

### 4. Train the model

```bash
python ai_training/train_model.py --dataset_root data --epochs 10
```

Saved outputs:
- `ai_training/output/best_model.h5`
- `ai_training/output/final_model.h5`
- `ai_training/output/classification_report.txt`

### 5. Convert to TensorFlow Lite

```bash
python tflite_model/convert_to_tflite.py --model ai_training/output/best_model.h5 --output tflite_model/drishti_model.tflite
```

### 6. Test one image offline

```bash
python test_scripts/test_tflite.py --model tflite_model/drishti_model.tflite --image data/val/normal/normal_val_1.png
# or let the script auto-pick the first validation image
python test_scripts/test_tflite.py --model tflite_model/drishti_model.tflite
```

### 7. Copy the generated model into Flutter assets

```bash
cp tflite_model/drishti_model.tflite flutter_app/assets/models/drishti_model.tflite
```

### 8. Run Flutter on Android

```bash
cd flutter_app
flutter create --platforms=android .
flutter pub get
flutter run
```

The repository currently ships the Flutter `lib/`, assets, and `pubspec.yaml`, but does not include the generated platform folders such as `android/`. Run `flutter create --platforms=android .` once inside `flutter_app/` to scaffold the Android runner before your first local build.

## Dataset layout

Store real images like this:

```text
data/
   train/
      glaucoma/
      normal/
   val/
      glaucoma/
      normal/
```

`dataset/` is still accepted as a fallback for backward compatibility, but all scripts now default to `data/`.

## Preprocessing steps

`preprocessing/preprocess.py` implements:
- Circular retina crop where possible
- Gaussian blur
- CLAHE contrast enhancement
- Resize to `224x224`
- Normalize pixels to `[0, 1]`

## Image quality check module

`quality_check/quality_checker.py` includes:
- Blur detection using Laplacian variance
- Brightness detection using HSV V-channel mean
- Center alignment scoring based on bright disc localization
- Final rejection logic with reasons

Example:

```bash
python quality_check/quality_checker.py --image data/val/normal/normal_val_1.png
```

## AI model training

`ai_training/train_model.py`:
- Uses `MobileNetV3Small` pretrained on ImageNet
- Freezes the backbone initially
- Adds `GlobalAveragePooling2D -> Dense(128, relu) -> Dropout(0.4) -> Dense(softmax)`
- Uses augmentation, inverse-frequency class weighting for imbalanced training sets, `EarlyStopping`, and `ModelCheckpoint`
- Trains for **at least 10 epochs**
- Refuses to train if duplicate images are present in both `train/` and `val/`, and reports a validation classification report without undefined-metric warnings

## Android deployment notes

- The MVP is designed for **offline inference only**.
- No cloud APIs are required.
- Use a smartphone + fundus lens attachment for acquisition.
- Replace the synthetic bootstrap dataset with validated clinical data before production deployment.
- Package a freshly trained `.tflite` model into `flutter_app/assets/models/drishti_model.tflite` before building a release APK/AAB.

## Notes and recommended next steps

- This scaffold currently uses two classes: `glaucoma` and `normal`.
- If you also want diabetic retinopathy, extend the dataset folders and labels to multi-class or multi-label screening.
- For stronger mobile quality checks, add vessel visibility, glare detection, and field-of-view estimation.
- For more aggressive size reduction under 5 MB, consider pruning or full int8 quantization with a representative dataset.


## Dependency compatibility notes

- The Python environment is pinned to `numpy<2` and `tensorflow<2.14` to avoid TensorFlow/NumPy ABI mismatches on Windows.
- Training checkpoints default to `.h5` output files for wider compatibility with TensorFlow 2.13.x on Windows.
- Use `scripts/split_dataset.py` when your source data starts in one folder per class, or run it with `--in-place` to repair an already-duplicated `data/` tree; it deduplicates by content and rebuilds stratified train/val splits.
