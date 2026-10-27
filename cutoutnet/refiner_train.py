"""Train the separate CutoutNet ROI CNN on high-resolution crops.

Corrupt ground-truth masks to approximate mistakes from a coarse segmentation.
For best results, supplement with *real* CutoutNet error masks via later fine-tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import random
import os

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .dataset import PairedCutoutDataset
from .model import ROIRefiner
from .losses import dice_loss


def make_noisy_mask(gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate realistic local errors including over-/under-segmentation and holes."""
    mask = (gt > 127).astype(np.uint8)
    h, w = mask.shape
    radius = int(rng.integers(1, 10))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    if rng.random() < .5:
        mask = cv2.dilate(mask, kernel)
    else:
        mask = cv2.erode(mask, kernel)
    if rng.random() < .5:
        dx, dy = rng.integers(-8, 9, size=2)
        affine = np.float32([[1, 0, int(dx)], [0, 1, int(dy)]])
        mask = cv2.warpAffine(mask, affine, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    if rng.random() < .65:
        for _ in range(int(rng.integers(1, 4))):
            cx, cy = int(rng.integers(w)), int(rng.integers(h))
            r = int(rng.integers(2, max(3, min(h, w) // 12)))
            color = int(rng.integers(0, 2))
            cv2.circle(mask, (cx, cy), r, color, thickness=-1)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), float(rng.uniform(.5, 2.0)))
    return np.clip(soft, .001, .999).astype(np.float32)


def uncertainty_from_coarse(p: np.ndarray) -> np.ndarray:
    hard = (p >= 0.5).astype(np.uint8)
    band = cv2.dilate(cv2.morphologyEx(hard, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)),
                     np.ones((7, 7), np.uint8))
    return np.maximum(1.0 - 2.0 * np.abs(p - .5), band.astype(np.float32)).astype(np.float32)


class RefinementDataset(Dataset):
    def __init__(self, root: str | Path, crop: int = 512, repeat: int = 4, seed: int = 2026, augment: bool = True):
        paired = PairedCutoutDataset(root, image_size=64, augment=False)
        self.pairs = paired.pairs
        self.crop = crop
        self.repeat = max(1, repeat) if augment else 1
        self.seed, self.augment = seed, augment

    def __len__(self):
        return len(self.pairs) * self.repeat

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        path_im, path_mask = self.pairs[idx % len(self.pairs)]
        rng = np.random.default_rng(self.seed + idx + (random.randrange(1000000) if self.augment else 0))
        with Image.open(path_im) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"), dtype=np.uint8).copy()
        with Image.open(path_mask) as im:
            mask = np.asarray(ImageOps.exif_transpose(im).convert("L"), dtype=np.uint8).copy()
        if mask.shape != rgb.shape[:2]:
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        h, w = mask.shape
        # Pick center near actual contour, thereby training precisely on hard regions.
        edge = cv2.morphologyEx((mask > 127).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
        ys, xs = np.where(edge > 0)
        if len(xs) and rng.random() < .8:
            k = int(rng.integers(len(xs)))
            cx, cy = int(xs[k]), int(ys[k])
        else:
            cx, cy = int(rng.integers(w)), int(rng.integers(h))
        s = self.crop
        x0 = max(0, min(w - s, cx - s // 2))
        y0 = max(0, min(h - s, cy - s // 2))
        rgb = rgb[y0:y0+s, x0:x0+s]
        mask = mask[y0:y0+s, x0:x0+s]
        # Constant padding keeps mask/background aligned for images smaller than crop.
        dh, dw = max(0, s-rgb.shape[0]), max(0, s-rgb.shape[1])
        rgb = cv2.copyMakeBorder(rgb, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        mask = cv2.copyMakeBorder(mask, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=0)
        if self.augment and rng.random() < .5:
            rgb, mask = rgb[:, ::-1].copy(), mask[:, ::-1].copy()
        if self.augment and rng.random() < .5:
            rgb, mask = rgb[::-1].copy(), mask[::-1].copy()
        if self.augment:
            noisy = make_noisy_mask(mask.copy(), rng)
        else:
            # Validation is deterministic and still measures correction performance.
            noisy = make_noisy_mask(mask.copy(), rng)
        uncert = uncertainty_from_coarse(noisy)
        return {
            "rgb": torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255.0,
            "coarse": torch.from_numpy(noisy.copy())[None],
            "uncertainty": torch.from_numpy(uncert.copy())[None],
            "gt": torch.from_numpy((mask > 127).astype(np.float32).copy())[None],
        }


def save_atomic(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".pt.tmp")
    torch.save(data, temp)
    os.replace(temp, path)


def train_refiner(train_root: str, val_root: str, out_dir: str = "checkpoints/refiner",
                  *, crop: int = 512, base: int = 48, batch_size: int = 2,
                  epochs: int = 20, lr: float = 1e-4, device: str = "auto",
                  workers: int = 0) -> None:
    device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    train_ds = RefinementDataset(train_root, crop=crop, augment=True)
    val_ds = RefinementDataset(val_root, crop=crop, augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=workers)
    model = ROIRefiner(base=base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    out = Path(out_dir)
    last = out / "last_full.pt"
    best, start = float("inf"), 1
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["base"] != base or state["crop"] != crop:
            raise ValueError("Refiner configuration differs from existing resume checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        best = float(state["best_val_loss"])
        start = int(state["epoch"]) + 1
    for epoch in range(start, epochs + 1):
        model.train()
        total = 0.0
        for batch in tqdm(train_loader, desc=f"ROI refiner train {epoch}/{epochs}"):
            data = {k: v.to(device) for k, v in batch.items()}
            logits0 = torch.logit(data["coarse"].clamp(1e-4, 1-1e-4))
            correction = model(data["rgb"], data["coarse"], data["uncertainty"])
            pred = logits0 + correction
            # Focus near disputed pixels while still supervising interiors.
            per_pixel = F.binary_cross_entropy_with_logits(pred.float(), data["gt"], reduction="none")
            weights = 1 + 2 * data["uncertainty"]
            loss = (per_pixel * weights).mean() + dice_loss(pred, data["gt"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        model.eval()
        val = 0.0
        with torch.inference_mode():
            for batch in val_loader:
                data = {k: v.to(device) for k, v in batch.items()}
                logit = torch.logit(data["coarse"].clamp(1e-4, 1-1e-4)) + model(data["rgb"], data["coarse"], data["uncertainty"])
                val += (F.binary_cross_entropy_with_logits(logit, data["gt"]) + dice_loss(logit, data["gt"])).item()
        score = val / len(val_loader)
        print(json.dumps({"epoch": epoch, "train": total / len(train_loader), "validation": score}))
        infer = {"format": "cutoutnet-refiner-v1", "model": model.state_dict(), "base": base, "crop": crop, "epoch": epoch}
        if score < best:
            best = score
            save_atomic(infer, out / "best.pt")
        save_atomic({**infer, "optimizer": optimizer.state_dict(), "best_val_loss": best}, last)
    print(f"Best trained ROI refiner: {out / 'best.pt'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train optional CutoutNet high-resolution ROI refiner")
    ap.add_argument("--train-root", default="data/cutoutnet/manifests/train.jsonl")
    ap.add_argument("--val-root", default="data/cutoutnet/manifests/val.jsonl")
    ap.add_argument("--output", default="checkpoints/refiner")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    train_refiner(args.train_root, args.val_root, args.output, crop=args.crop, base=args.base,
                  epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                  device=args.device, workers=args.workers)


if __name__ == "__main__":
    main()
