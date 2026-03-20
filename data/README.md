# Data directory

This repository now uses `data/` as the default dataset root.

Expected layout:

```text
data/
├── train/
│   ├── glaucoma/
│   └── normal/
└── val/
    ├── glaucoma/
    └── normal/
```

Run `python scripts/bootstrap_project.py --skip-install` to create a small synthetic sample dataset for smoke tests, or run it without `--skip-install` to also create `.venv/` and install Python dependencies.

The generated images are intentionally ignored by git so pull requests stay text-only and avoid binary attachment issues.
