from __future__ import annotations

import argparse
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def files(path: Path):
    return [p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS]


def random_background(rng: random.Random, size: int, background_files: list[Path]) -> np.ndarray:
    mode = rng.choice(["photo", "solid", "gradient", "white", "black", "chroma"])
    if mode == "photo" and background_files:
        p = rng.choice(background_files)
        rgb = np.asarray(Image.open(p).convert("RGB"), np.uint8)
        h, w = rgb.shape[:2]
        scale = max(size / max(1, h), size / max(1, w))
        nw, nh = max(size, int(w * scale)), max(size, int(h * scale))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_CUBIC)
        x = rng.randint(0, max(0, nw - size)); y = rng.randint(0, max(0, nh - size))
        return rgb[y:y+size, x:x+size].copy()
    if mode == "white":
        return np.full((size, size, 3), rng.randint(242, 255), np.uint8)
    if mode == "black":
        return np.full((size, size, 3), rng.randint(0, 18), np.uint8)
    if mode == "chroma":
        choices = [(145, 238, 78), (35, 210, 65), (35, 85, 225), (180, 40, 205)]
        base = np.array(rng.choice(choices), np.int16)
        jitter = np.random.default_rng(rng.randrange(1 << 30)).normal(0, 2.0, (size, size, 3))
        return np.clip(base + jitter, 0, 255).astype(np.uint8)
    if mode == "gradient":
        a = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
        b = np.array([rng.randint(0, 255) for _ in range(3)], np.float32)
        t = np.linspace(0, 1, size, dtype=np.float32)[:, None, None]
        return np.repeat((a[None, None] * (1 - t) + b[None, None] * t), size, axis=1).astype(np.uint8)
    color = np.array([rng.randint(0, 255) for _ in range(3)], np.uint8)
    return np.tile(color[None, None], (size, size, 1))


def transform_rgba(rgba: np.ndarray, rng: random.Random, canvas: int):
    h, w = rgba.shape[:2]
    target = rng.randint(max(32, canvas // 9), max(64, int(canvas * 0.70)))
    scale = target / max(h, w)
    nw, nh = max(2, int(w * scale)), max(2, int(h * scale))
    rgba = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    angle = rng.uniform(-24, 24)
    center = (nw / 2, nh / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    bw, bh = int(nh * sin + nw * cos), int(nh * cos + nw * sin)
    M[0, 2] += bw / 2 - center[0]; M[1, 2] += bh / 2 - center[1]
    rgba = cv2.warpAffine(rgba, M, (bw, bh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return rgba


def composite(bg: np.ndarray, rgba: np.ndarray, x: int, y: int, mask: np.ndarray):
    h, w = rgba.shape[:2]
    H, W = bg.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    rx0, ry0 = x0 - x, y0 - y
    rx1, ry1 = rx0 + (x1 - x0), ry0 + (y1 - y0)
    patch = rgba[ry0:ry1, rx0:rx1]
    alpha = patch[..., 3:4].astype(np.float32) / 255.0
    # Deliberately preserve source edge matting in RGB; GT remains binary.
    bg[y0:y1, x0:x1] = np.clip(patch[..., :3] * alpha + bg[y0:y1, x0:x1] * (1 - alpha), 0, 255).astype(np.uint8)
    mask[y0:y1, x0:x1][patch[..., 3] >= 128] = 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutouts", default="data/cutoutnet/cutouts")
    ap.add_argument("--backgrounds", default="data/cutoutnet/backgrounds")
    ap.add_argument("--out", default="data/cutoutnet/synthetic")
    ap.add_argument("--count", type=int, default=10000)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--max-objects", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    cutouts = files(Path(args.cutouts)); backgrounds = files(Path(args.backgrounds))
    if not cutouts:
        raise SystemExit("No transparent PNG cutouts found. Put RGBA files in data/cutoutnet/cutouts/")
    out = Path(args.out); (out / "images").mkdir(parents=True, exist_ok=True); (out / "masks").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for i in range(args.count):
        bg = random_background(rng, args.size, backgrounds)
        mask = np.zeros((args.size, args.size), np.uint8)
        nobj = rng.randint(1, args.max_objects)
        for _ in range(nobj):
            p = rng.choice(cutouts)
            rgba = np.asarray(Image.open(p).convert("RGBA"), np.uint8)
            if np.max(rgba[..., 3]) == 0:
                continue
            rgba = transform_rgba(rgba, rng, args.size)
            h, w = rgba.shape[:2]
            # Sometimes allow objects to touch/cross the frame edge.
            x = rng.randint(-w // 5, max(-w // 5, args.size - max(1, w * 4 // 5)))
            y = rng.randint(-h // 5, max(-h // 5, args.size - max(1, h * 4 // 5)))
            composite(bg, rgba, x, y, mask)
        name = f"syn_{i:07d}"
        Image.fromarray(bg).save(out / "images" / f"{name}.jpg", quality=92)
        Image.fromarray(mask).save(out / "masks" / f"{name}.png")
        if (i + 1) % 100 == 0 or i + 1 == args.count:
            print(f"{i+1}/{args.count}")


if __name__ == "__main__":
    main()
