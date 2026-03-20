"""Bootstrap the DRISHTI repository for local runs and deployment packaging."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import venv
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import binascii
import struct
import zlib

from project_paths import ensure_dataset_layout, repo_root

IMAGE_SIZE = (224, 224)
SAMPLES_PER_CLASS = 2


def run_command(command: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess command and stream output."""
    print(f"$ {' '.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def create_virtualenv(venv_path: Path) -> Path:
    """Create a virtualenv if missing and return the python executable path."""
    if not venv_path.exists():
        print(f'Creating virtual environment at {venv_path}')
        venv.EnvBuilder(with_pip=True).create(venv_path)
    scripts_dir = 'Scripts' if platform.system() == 'Windows' else 'bin'
    python_path = venv_path / scripts_dir / ('python.exe' if platform.system() == 'Windows' else 'python')
    if not python_path.exists():
        raise FileNotFoundError(f'Virtualenv python not found at {python_path}')
    return python_path


def install_requirements(python_path: Path, requirements_path: Path) -> None:
    """Install Python dependencies into the virtualenv."""
    run_command([str(python_path), '-m', 'pip', 'install', '--upgrade', 'pip'])
    run_command([str(python_path), '-m', 'pip', 'install', '-r', str(requirements_path)])


def _empty_canvas() -> list[list[list[int]]]:
    return [[[10, 10, 10] for _ in range(IMAGE_SIZE[0])] for _ in range(IMAGE_SIZE[1])]


def _fill_circle(canvas: list[list[list[int]]], center_x: int, center_y: int, radius: int, color: tuple[int, int, int]) -> None:
    for y in range(max(0, center_y - radius), min(IMAGE_SIZE[1], center_y + radius)):
        for x in range(max(0, center_x - radius), min(IMAGE_SIZE[0], center_x + radius)):
            if (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2:
                canvas[y][x] = list(color)


def _draw_vessel(canvas: list[list[list[int]]], points: list[tuple[int, int]], color: tuple[int, int, int], width: int = 4) -> None:
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        steps = max(abs(x2 - x1), abs(y2 - y1), 1)
        for step in range(steps + 1):
            x = round(x1 + (x2 - x1) * step / steps)
            y = round(y1 + (y2 - y1) * step / steps)
            _fill_circle(canvas, x, y, width, color)


def _box_blur(canvas: list[list[list[int]]], radius: int) -> list[list[list[int]]]:
    if radius <= 0:
        return canvas
    result = _empty_canvas()
    for y in range(IMAGE_SIZE[1]):
        for x in range(IMAGE_SIZE[0]):
            pixels: list[list[int]] = []
            for ky in range(max(0, y - radius), min(IMAGE_SIZE[1], y + radius + 1)):
                for kx in range(max(0, x - radius), min(IMAGE_SIZE[0], x + radius + 1)):
                    pixels.append(canvas[ky][kx])
            result[y][x] = [sum(channel[idx] for channel in pixels) // len(pixels) for idx in range(3)]
    return result


def _write_png(output_path: Path, canvas: list[list[list[int]]]) -> None:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack('!I', len(data)) + tag + data + struct.pack('!I', binascii.crc32(tag + data) & 0xFFFFFFFF)

    raw = b''.join(b'\x00' + bytes(pixel for rgb in row for pixel in rgb) for row in canvas)
    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('!2I5B', IMAGE_SIZE[0], IMAGE_SIZE[1], 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(raw, level=9))
    png += chunk(b'IEND', b'')
    output_path.write_bytes(png)


def create_sample_image(output_path: Path, label: str, variant: int) -> None:
    """Generate a synthetic but retina-like sample image."""
    canvas = _empty_canvas()

    if label == 'normal':
        _fill_circle(canvas, 112, 112, 96, (186, 68 + variant * 8, 58))
        _fill_circle(canvas, 152 + variant * 2, 108, 20, (242, 220, 150))
        vessel_color = (110, 30, 26)
        blur_radius = 1
    else:
        _fill_circle(canvas, 112, 112, 96, (122, 86, 54 + variant * 6))
        _fill_circle(canvas, 170, 100 + variant, 22, (242, 220, 150))
        vessel_color = (160, 120, 80)
        for ring in range(44, 53):
            for angle_step in range(360):
                from math import cos, radians, sin
                angle = radians(angle_step)
                x = int(112 + ring * cos(angle))
                y = int(112 + ring * sin(angle))
                if 0 <= x < IMAGE_SIZE[0] and 0 <= y < IMAGE_SIZE[1]:
                    canvas[y][x] = [236, 206, 132]
        blur_radius = 2

    vessel_paths = [
        [(110, 112), (74, 72), (46, 54)],
        [(112, 114), (88, 148), (72, 180)],
        [(118, 110), (160, 70), (186, 54)],
        [(122, 118), (154, 156), (176, 184)],
    ]
    for vessel in vessel_paths:
        _draw_vessel(canvas, vessel, vessel_color, width=4)

    blurred = _box_blur(canvas, blur_radius)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_png(output_path, blurred)


def populate_sample_dataset(data_root: Path, force: bool = False) -> list[str]:
    """Populate the data directory with synthetic sample images when empty."""
    created: list[str] = []
    for split in ['train', 'val']:
        for label in ['glaucoma', 'normal']:
            class_dir = data_root / split / label
            existing_images = [path for path in class_dir.iterdir() if path.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
            if existing_images and not force:
                continue
            for idx in range(SAMPLES_PER_CLASS):
                filename = f'{label}_{split}_{idx + 1}.png'
                output_path = class_dir / filename
                create_sample_image(output_path, label, idx)
                created.append(str(output_path.relative_to(repo_root())))
    return created


def write_manifest(data_root: Path, created_files: Iterable[str]) -> Path:
    """Write a small manifest describing the generated sample dataset."""
    manifest_path = data_root / 'sample_manifest.json'
    payload = {
        'generated_by': 'scripts/bootstrap_project.py',
        'python': sys.version.split()[0],
        'files': sorted(created_files),
        'note': 'Synthetic sample images for smoke tests only; replace with validated clinical data before deployment.',
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + '\n')
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Bootstrap DRISHTI for local execution.')
    parser.add_argument('--venv', default='.venv', help='Virtual environment path to create/update')
    parser.add_argument('--skip-install', action='store_true', help='Skip pip dependency installation')
    parser.add_argument('--force-samples', action='store_true', help='Overwrite existing sample images')
    args = parser.parse_args()

    root = repo_root()
    data_root = ensure_dataset_layout(root / 'data')
    created_files = populate_sample_dataset(data_root, force=args.force_samples)
    manifest_path = write_manifest(data_root, created_files)
    print(f'Data directory ready at {data_root}')
    print(f'Manifest written to {manifest_path}')

    python_path: Path | None = None
    if not args.skip_install:
        python_path = create_virtualenv(root / args.venv)
        install_requirements(python_path, root / 'requirements.txt')
    else:
        print('Skipping dependency installation by request.')

    print('\nBootstrap complete.')
    print(f'- Dataset root: {data_root}')
    print(f'- Sample images created: {len(created_files)}')
    if python_path is not None:
        print(f'- Virtualenv python: {python_path}')
        print(f'- Activate with: source {args.venv}/bin/activate')
    print('- Next: run python ai_training/train_model.py --dataset_root data --epochs 10')


if __name__ == '__main__':
    main()
