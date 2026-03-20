# Bundled demo dataset

This repository ships with a **small synthetic retinal dataset generator** so DRISHTI remains runnable immediately without committing binary image files.

## Generated samples

When generated, the demo dataset contains:

- `train/glaucoma`: 8 demo images
- `train/normal`: 8 demo images
- `val/glaucoma`: 4 demo images
- `val/normal`: 4 demo images

These images are **procedurally generated development assets** intended only for testing the preprocessing, training, conversion, and app wiring. They are **not clinically valid** and must be replaced with real validated fundus datasets before any meaningful medical screening work.

## Regenerate

```bash
python dataset/generate_demo_dataset.py
```

The training script will auto-generate this dataset if the folders are empty.
