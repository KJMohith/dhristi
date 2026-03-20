# DRISHTI MVP

DRISHTI is an offline smartphone-based retinal fundus screening MVP designed for low-cost Android deployment. This repository includes Python tools for preprocessing, data quality checks, model training, TFLite conversion, and a Flutter app that performs on-device inference without cloud connectivity.

## Final Project Structure

```text
DRISHTI/
├── ai_training/
│   └── train_model.py
├── dataset/
│   ├── train/
│   │   ├── glaucoma/
│   │   └── normal/
│   └── val/
│       ├── glaucoma/
│       └── normal/
├── flutter_app/
│   ├── assets/models/
│   │   └── drishti_model.tflite
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
├── test_scripts/
│   └── test_tflite.py
├── tflite_model/
│   └── convert_to_tflite.py
├── README.md
└── requirements.txt
```

## Part 1 — Data Pipeline

### Dataset Layout

Store images like this:

```text
dataset/
   train/
      glaucoma/
      normal/
   val/
      glaucoma/
      normal/
```

### Preprocessing Steps

`preprocessing/preprocess.py` implements:
- Circular retina crop where possible
- Gaussian blur
- CLAHE contrast enhancement
- Resize to `224x224`
- Normalize pixels to `[0, 1]`

### Example Commands

```bash
python preprocessing/preprocess.py --input dataset/train --output processed/train
python preprocessing/dataset_loader.py --dataset_root dataset
```

## Part 2 — Image Quality Check Module

`quality_check/quality_checker.py` includes:
- Blur detection using Laplacian variance
- Brightness detection using HSV V-channel mean
- Center alignment scoring based on bright disc localization
- Final rejection logic with reasons

Example:

```bash
python quality_check/quality_checker.py --image sample_fundus.jpg
```

## Part 3 — AI Model Training

`ai_training/train_model.py`:
- Uses `MobileNetV3Small` pretrained on ImageNet
- Freezes the backbone initially
- Adds `GlobalAveragePooling2D -> Dense(128, relu) -> Dropout(0.4) -> Dense(softmax)`
- Uses augmentation, `EarlyStopping`, and `ModelCheckpoint`
- Trains for **at least 10 epochs**
- Reports accuracy, precision, and recall during training, plus a validation classification report at the end

Train with:

```bash
python ai_training/train_model.py --dataset_root dataset --epochs 10
```

Saved outputs:
- `ai_training/output/best_model.keras`
- `ai_training/output/final_model.keras`
- `ai_training/output/classification_report.txt`

## Part 4 — Model Compression

Convert to TensorFlow Lite with post-training quantization:

```bash
python tflite_model/convert_to_tflite.py --model ai_training/output/best_model.keras --output tflite_model/drishti_model.tflite
```

The converter applies `tf.lite.Optimize.DEFAULT` with float16 quantization. The script prints the final model size and warns if it is above the `< 5 MB` target.

## Part 5 — TFLite Test Script

`test_scripts/test_tflite.py` loads a `.tflite` model, preprocesses an input fundus image, runs inference, and prints:
- Predicted class
- Confidence score
- Raw probability vector

Usage:

```bash
python test_scripts/test_tflite.py --model tflite_model/drishti_model.tflite --image sample_fundus.jpg
```

## Part 6 + 7 — Flutter Mobile App + TFLite Integration

The Flutter app in `flutter_app/` includes:

- **Camera Capture Screen**
  - Live camera preview
  - Circular alignment overlay
  - Capture and analyze button
  - On-device quality gate before inference
- **AI Result Screen**
  - Traffic-light triage
    - Green → Healthy
    - Yellow → Risk
    - Red → Refer doctor
- **History Screen**
  - Stores scans locally with `SharedPreferences`
- **TFLite Integration**
  - Loads `assets/models/drishti_model.tflite`
  - Resizes input to `224x224`
  - Normalizes pixels to `[0, 1]`
  - Runs offline inference with `tflite_flutter`

### Flutter Setup

1. Install Flutter and Android Studio.
2. Copy your generated TFLite model into:
   - `flutter_app/assets/models/drishti_model.tflite`
3. Fetch packages:

```bash
cd flutter_app
flutter pub get
```

4. Run on Android:

```bash
flutter run
```

## Android Deployment Notes

- The MVP is designed for **offline inference only**.
- No cloud APIs are required.
- Use a smartphone + fundus lens attachment for acquisition.
- For real clinical deployment, add stronger validation, calibration, and regulatory workflows.

## Step-by-Step End-to-End Run Instructions

1. **Create a Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Add dataset images** under `dataset/train/...` and `dataset/val/...`.
3. **Optionally preprocess** the dataset:
   ```bash
   python preprocessing/preprocess.py --input dataset/train --output processed/train
   python preprocessing/preprocess.py --input dataset/val --output processed/val
   ```
4. **Train the model**
   ```bash
   python ai_training/train_model.py --dataset_root dataset --epochs 10
   ```
5. **Convert to TFLite**
   ```bash
   python tflite_model/convert_to_tflite.py --model ai_training/output/best_model.keras
   ```
6. **Test one image offline**
   ```bash
   python test_scripts/test_tflite.py --model tflite_model/drishti_model.tflite --image sample_fundus.jpg
   ```
7. **Copy model into Flutter assets**
   ```bash
   cp tflite_model/drishti_model.tflite flutter_app/assets/models/drishti_model.tflite
   ```
8. **Run Flutter app on Android**
   ```bash
   cd flutter_app
   flutter pub get
   flutter run
   ```

## Notes and Recommended Next Steps

- This scaffold currently uses two classes from the dataset structure you specified: `glaucoma` and `normal`.
- If you also want diabetic retinopathy, extend the dataset folders and labels to multi-class or multi-label screening.
- For stronger mobile quality checks, add vessel visibility, glare detection, and field-of-view estimation.
- For more aggressive size reduction under 5 MB, consider pruning or full int8 quantization with a representative dataset.
