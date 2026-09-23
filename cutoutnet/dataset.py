from __future__ import annotations

from pathlib import Path
import random
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance

from .targets import build_targets


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _resolve_manifest_path(raw: str, project_root: Path) -> Path:
    """Resolve manifest paths across Windows/local/Colab relocations.

    New manifests use project-relative paths, but older manifests may contain a
    Windows absolute path or an ephemeral ``/content/...`` path created by a
    previous Colab session.  When that original path is gone, remap the stable
    dataset suffix under the current project's data/ tree.
    """
    raw_s = str(raw)
    normalized = raw_s.replace("\\", "/")

    direct = Path(raw_s)
    if direct.is_absolute() and direct.exists():
        return direct
    if not direct.is_absolute():
        candidate = project_root / direct
        if candidate.exists():
            return candidate

    # Portable anchors used by CutoutNet's dataset manager.
    for anchor in ("data/raw/", "data/cutoutnet/"):
        pos = normalized.lower().find(anchor)
        if pos >= 0:
            candidate = project_root / normalized[pos:]
            if candidate.exists():
                return candidate

    # Legacy cloud manifests sometimes resolved the data/raw/P3M-10k symlink and
    # stored /content/.../P3M-10k/... instead. Reattach to the current symlink.
    for dataset_name in ("P3M-10k", "AM-2K", "AIM-500"):
        token = f"/{dataset_name}/"
        pos = normalized.lower().find(token.lower())
        if pos >= 0:
            suffix = normalized[pos + len(token):]
            candidate = project_root / "data" / "raw" / dataset_name / suffix
            if candidate.exists():
                return candidate

    return project_root / direct


def _match_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    masks = {p.stem: p for p in mask_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    pairs = []
    for img in sorted(image_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        mask = masks.get(img.stem)
        if mask is not None:
            pairs.append((img, mask))
    return pairs


def _letterbox_pair(rgb: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = rgb.shape[:2]
    scale = min(size / max(1, w), size / max(1, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    rgb_r = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    mask_r = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size, 3), np.uint8)
    mask_canvas = np.zeros((size, size), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = rgb_r
    mask_canvas[y0:y0+nh, x0:x0+nw] = mask_r
    return canvas, mask_canvas


def _augment(rgb: np.ndarray, mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        rgb = np.ascontiguousarray(rgb[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
    if rng.random() < 0.12:
        rgb = np.ascontiguousarray(rgb[::-1])
        mask = np.ascontiguousarray(mask[::-1])

    pil = Image.fromarray(rgb)
    if rng.random() < 0.7:
        pil = ImageEnhance.Brightness(pil).enhance(rng.uniform(0.82, 1.18))
    if rng.random() < 0.7:
        pil = ImageEnhance.Contrast(pil).enhance(rng.uniform(0.82, 1.22))
    if rng.random() < 0.55:
        pil = ImageEnhance.Color(pil).enhance(rng.uniform(0.70, 1.30))
    rgb = np.asarray(pil, dtype=np.uint8)

    # Mild JPEG-like blur/noise makes edge learning less dependent on pristine masks.
    if rng.random() < 0.18:
        rgb = cv2.GaussianBlur(rgb, (3, 3), rng.uniform(0.25, 0.9))
    if rng.random() < 0.20:
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(0.0, rng.uniform(1.0, 4.0), rgb.shape)
        rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return rgb, mask


class PairedCutoutDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        image_size: int = 768,
        augment: bool = True,
        target_cfg: dict | None = None,
        seed: int = 1337,
    ):
        self.root = Path(root)
        if self.root.suffix.lower() == ".jsonl":
            if not self.root.exists():
                raise FileNotFoundError(self.root)
            project_root = Path.cwd()
            self.pairs = []
            for line in self.root.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                img = _resolve_manifest_path(row["image"], project_root)
                mask = _resolve_manifest_path(row["mask"], project_root)
                if img.exists() and mask.exists():
                    self.pairs.append((img, mask))
        else:
            self.image_dir = self.root / "images"
            self.mask_dir = self.root / "masks"
            if not self.image_dir.exists() or not self.mask_dir.exists():
                raise FileNotFoundError(f"Expected {self.image_dir} and {self.mask_dir}")
            self.pairs = _match_pairs(self.image_dir, self.mask_dir)
        if not self.pairs:
            raise RuntimeError(f"No matching image/mask pairs found under {self.root}")
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.target_cfg = target_cfg or {}
        self.seed = int(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        img_path, mask_path = self.pairs[index]
        rgb = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8, copy=True)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8, copy=True)
        rgb, mask = _letterbox_pair(rgb, mask, self.image_size)
        if self.augment:
            rng = random.Random(self.seed + index + random.randrange(1 << 20))
            rgb, mask = _augment(rgb, mask, rng)

        targets_np = build_targets(mask, self.target_cfg)
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
        # ImageNet normalization keeps pretrained encoders useful.
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        image = (image - mean) / std
        targets = {
            k: torch.from_numpy(np.ascontiguousarray(v)).unsqueeze(0).float()
            for k, v in targets_np.items()
        }
        return {
            "image": image,
            "targets": targets,
            "name": img_path.stem,
        }
