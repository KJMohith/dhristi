# Flutter deployment quick start

This folder contains the DRISHTI Flutter application source, assets, and package manifest.

## First-time Android setup

The repository intentionally keeps the Flutter app lightweight and does **not** commit generated platform folders such as `android/`.

From this directory, run:

```bash
flutter create --platforms=android .
flutter pub get
```

Then confirm that the offline model is bundled:

```bash
ls assets/models/drishti_model.tflite
```

## Run on an Android device

```bash
flutter run
```

## Release build

```bash
flutter build apk --release
```

## Notes

- The app expects `assets/models/drishti_model.tflite` to exist before runtime.
- Camera permission handling is provided by the `camera` plugin after the Android project is scaffolded.
- Gallery imports use `image_picker`, so the same offline quality gate and TFLite analysis also work with existing photos.
- Scan history is stored locally with `shared_preferences`, so the MVP works offline.
