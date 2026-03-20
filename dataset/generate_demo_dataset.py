"""Generate a tiny synthetic retinal dataset so the DRISHTI repo is runnable immediately.

This uses only the Python standard library so it can run in constrained environments.
The generated images are development/demo assets only and are not clinically valid.
"""
from __future__ import annotations

import math
import random
import struct
from pathlib import Path

SPLITS = {'train': 8, 'val': 4}
CLASSES = ['glaucoma', 'normal']
ROOT = Path(__file__).resolve().parent
SIZE = 256


def _clamp(value: float) -> int:
    return max(0, min(255, int(value)))


def _write_bmp(path: Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    """Write a 24-bit BMP using only stdlib."""
    height = len(pixels)
    width = len(pixels[0])
    row_padding = (4 - (width * 3) % 4) % 4
    pixel_bytes = bytearray()

    for row in reversed(pixels):
        for red, green, blue in row:
            pixel_bytes.extend(bytes((blue, green, red)))
        pixel_bytes.extend(b'\x00' * row_padding)

    file_size = 14 + 40 + len(pixel_bytes)
    header = struct.pack('<2sIHHI', b'BM', file_size, 0, 0, 54)
    dib = struct.pack('<IIIHHIIIIII', 40, width, height, 1, 24, 0, len(pixel_bytes), 2835, 2835, 0, 0)
    path.write_bytes(header + dib + pixel_bytes)


def _draw_circle(pixels, cx, cy, radius, color):
    radius_sq = radius * radius
    y_min = max(0, cy - radius)
    y_max = min(SIZE, cy + radius)
    x_min = max(0, cx - radius)
    x_max = min(SIZE, cx + radius)
    for y in range(y_min, y_max):
        dy = y - cy
        for x in range(x_min, x_max):
            dx = x - cx
            if dx * dx + dy * dy <= radius_sq:
                pixels[y][x] = color


def _blend(pixel, overlay, alpha):
    return tuple(_clamp((1 - alpha) * pixel[i] + alpha * overlay[i]) for i in range(3))


def make_fundus(label: str, seed: int) -> list[list[tuple[int, int, int]]]:
    rng = random.Random(seed)
    pixels = [[(0, 0, 0) for _ in range(SIZE)] for _ in range(SIZE)]
    cx = SIZE // 2 + rng.randint(-8, 8)
    cy = SIZE // 2 + rng.randint(-8, 8)
    radius = rng.randint(92, 106)

    for y in range(SIZE):
        for x in range(SIZE):
            dx = x - cx
            dy = y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= radius:
                norm = dist / radius
                red = 32 + int(80 * (1 - norm))
                green = 56 + int(92 * (1 - norm))
                blue = 90 + int(80 * (1 - norm))
                vignette = max(0.25, 1.08 - norm * 0.92)
                pixels[y][x] = (_clamp(red * vignette), _clamp(green * vignette), _clamp(blue * vignette))

    vessel_count = 32 if label == 'glaucoma' else 24
    vessel_color = (88, 26, 24)
    for _ in range(vessel_count):
        angle = rng.random() * math.tau
        steps = rng.randint(32, 60)
        branch_origin = (cx + rng.randint(-10, 10), cy + rng.randint(-10, 10))
        prev_x, prev_y = branch_origin
        for step in range(1, steps):
            distance = step * (radius / steps)
            bend = math.sin(step / 6) * rng.uniform(-0.25, 0.25)
            x = int(branch_origin[0] + math.cos(angle + bend) * distance)
            y = int(branch_origin[1] + math.sin(angle + bend) * distance)
            thickness = 2 if step < steps * 0.35 else 1
            for offset_y in range(-thickness, thickness + 1):
                for offset_x in range(-thickness, thickness + 1):
                    px = x + offset_x
                    py = y + offset_y
                    if 0 <= px < SIZE and 0 <= py < SIZE:
                        if (px - cx) ** 2 + (py - cy) ** 2 <= radius * radius:
                            pixels[py][px] = vessel_color
            prev_x, prev_y = x, y

    disc_x = cx + rng.randint(28, 52)
    disc_y = cy + rng.randint(-18, 18)
    disc_radius = rng.randint(18, 24)
    cup_radius = int(disc_radius * (0.72 if label == 'glaucoma' else 0.44))
    _draw_circle(pixels, disc_x, disc_y, disc_radius, (238, 208, 164))
    _draw_circle(pixels, disc_x, disc_y, max(cup_radius, 8), (250, 234, 208))

    lesion_count = 18 if label == 'glaucoma' else 8
    for _ in range(lesion_count):
        lx = rng.randint(cx - radius // 2, cx + radius // 2)
        ly = rng.randint(cy - radius // 2, cy + radius // 2)
        if (lx - cx) ** 2 + (ly - cy) ** 2 > radius * radius:
            continue
        lesion_radius = rng.randint(2, 4 if label == 'glaucoma' else 3)
        lesion_color = (44, 36, 96) if label == 'glaucoma' else (78, 92, 142)
        _draw_circle(pixels, lx, ly, lesion_radius, lesion_color)

    for y in range(SIZE):
        for x in range(SIZE):
            if pixels[y][x] == (0, 0, 0):
                continue
            noise = rng.randint(-8, 8)
            pixels[y][x] = tuple(_clamp(channel + noise) for channel in pixels[y][x])
    return pixels


def generate_dataset() -> None:
    for split, count in SPLITS.items():
        for class_name in CLASSES:
            output_dir = ROOT / split / class_name
            output_dir.mkdir(parents=True, exist_ok=True)
            for image_path in output_dir.glob('*'):
                if image_path.is_file():
                    image_path.unlink()
            for index in range(count):
                pixels = make_fundus(class_name, seed=1000 + index + (0 if split == 'train' else 100) + (0 if class_name == 'normal' else 1000))
                _write_bmp(output_dir / f'{class_name}_{index + 1:02d}.bmp', pixels)


if __name__ == '__main__':
    generate_dataset()
    print('Synthetic demo dataset regenerated successfully.')
