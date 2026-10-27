"""End-to-end CutoutNet inference with optional trained high-resolution ROI CNN.

Backwards compatible with CutoutNet v0.2.4 `best.pt` and `last_full.pt`.
No pretrained weights are downloaded: checkpoints are mandatory for inference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch

from .model import CutoutNet, ROIRefiner
from .validator import binary_alpha, risk_map, select_rois, ROI


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_device(choice: str = "auto") -> torch.device:
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(choice)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is unavailable")
    return device


def read_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode == "RGBA":
            # Alpha is not treated as a GT mask; composite for RGB inference.
            bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
            im = Image.alpha_composite(bg, im)
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


@dataclass(frozen=True)
class Letterbox:
    side: int
    x0: int
    y0: int
    width: int
    height: int
    original_size: tuple[int, int]  # h,w


def make_letterbox(rgb: np.ndarray, side: int) -> tuple[np.ndarray, Letterbox]:
    if side < 32:
        raise ValueError("side must be at least 32")
    h, w = rgb.shape[:2]
    scale = min(side / h, side / w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    reduced = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((side, side, 3), np.uint8)  # matches v0.2 training exactly
    x0, y0 = (side - nw) // 2, (side - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = reduced
    return canvas, Letterbox(side, x0, y0, nw, nh, (h, w))


def unletterbox(p: np.ndarray, info: Letterbox) -> np.ndarray:
    cropped = p[info.y0:info.y0+info.height, info.x0:info.x0+info.width]
    h, w = info.original_size
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def weighted_refine(prob: np.ndarray, uncertainty: np.ndarray, rgb: np.ndarray,
                    regions: list[ROI], model: ROIRefiner, device: torch.device) -> np.ndarray:
    """Use v0.2 ROIRefiner's residual-logit contract; blend overlapping tiles.

    Pixels outside selected tiles are left exactly unchanged.
    """
    if not regions:
        return prob
    h, w = prob.shape
    accumulated = np.zeros((h, w), np.float32)
    weights = np.zeros((h, w), np.float32)
    model.eval()
    with torch.inference_mode():
        for roi in regions:
            x0, y0, x1, y1 = roi.box
            p = prob[y0:y1, x0:x1]
            u = uncertainty[y0:y1, x0:x1]
            rgb_t = torch.from_numpy(np.ascontiguousarray(rgb[y0:y1, x0:x1].transpose(2, 0, 1))).float()[None].to(device) / 255.0
            p_t = torch.from_numpy(np.ascontiguousarray(p))[None, None].to(device)
            u_t = torch.from_numpy(np.ascontiguousarray(u))[None, None].to(device)
            # Refiner output is an additive *logit correction*.
            delta = model(rgb_t, p_t, u_t).float().clamp(-10.0, 10.0)
            logits = torch.logit(p_t.clamp(1e-4, 1-1e-4))
            corrected = torch.sigmoid(logits + delta).squeeze().cpu().numpy()
            corrected = np.atleast_2d(corrected).astype(np.float32)
            ph, pw = p.shape
            # Avoid seams while retaining strictly positive weights on all borders.
            wy = np.hanning(ph) if ph > 2 else np.ones(ph)
            wx = np.hanning(pw) if pw > 2 else np.ones(pw)
            window = np.maximum(np.outer(wy, wx).astype(np.float32), 0.03)
            accumulated[y0:y1, x0:x1] += window * corrected
            weights[y0:y1, x0:x1] += window
    output = prob.copy()
    modified = weights > 0
    output[modified] = accumulated[modified] / weights[modified]
    return output


@dataclass
class Result:
    rgb: np.ndarray
    alpha: np.ndarray
    probability: np.ndarray
    risk: np.ndarray
    regions: list[ROI]
    checkpoint_epoch: int | None
    refiner_used: bool

    def rgba(self) -> np.ndarray:
        rgba = np.dstack((self.rgb.copy(), self.alpha))
        rgba[self.alpha == 0, :3] = 0  # prevent hidden original background RGB
        return rgba

    def save(self, path: str | Path, save_diagnostics: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.rgba(), "RGBA").save(path)
        if save_diagnostics:
            Image.fromarray(self.alpha).save(path.with_name(path.stem + "_mask.png"))
            Image.fromarray((np.clip(self.risk, 0, 1) * 255).astype(np.uint8)).save(path.with_name(path.stem + "_risk.png"))
            diagnostic = {
                "checkpoint_epoch": self.checkpoint_epoch,
                "refiner_used": self.refiner_used,
                "roi_count": len(self.regions),
                "roi_boxes": [r.box for r in self.regions],
                "image_size": [int(self.rgb.shape[1]), int(self.rgb.shape[0])],
            }
            path.with_name(path.stem + "_diagnostics.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")


class CutoutPipeline:
    def __init__(self, checkpoint: str | Path, *, refiner_checkpoint: str | Path | None = None,
                 device: str = "auto", image_size: int | None = None, threshold: float = 0.5,
                 patch: int = 512, overlap: int = 128, max_rois: int = 80):
        self.device = resolve_device(device)
        self.threshold = threshold
        self.patch, self.overlap, self.max_rois = patch, overlap, max_rois
        try:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            raise FileNotFoundError(f"Trained CutoutNet weights not found: {checkpoint}") from None
        if not isinstance(ckpt, dict) or "model" not in ckpt or "model_config" not in ckpt:
            raise ValueError("Expected v0.2 CutoutNet checkpoint with model/model_config")
        cfg = dict(ckpt["model_config"])
        cfg["pretrained_backbone"] = False  # offline, no fresh model download
        self.model = CutoutNet(**cfg)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.to(self.device).eval()
        self.image_size = int(image_size or ckpt.get("image_size", 640))
        self.epoch = ckpt.get("epoch")
        self.refiner = None
        if refiner_checkpoint is not None:
            ref_ckpt = torch.load(refiner_checkpoint, map_location="cpu", weights_only=False)
            if not isinstance(ref_ckpt, dict) or ref_ckpt.get("format") != "cutoutnet-refiner-v1":
                raise ValueError("Unrecognized ROI refiner checkpoint; refusing to use random weights")
            self.refiner = ROIRefiner(base=int(ref_ckpt["base"]))
            self.refiner.load_state_dict(ref_ckpt["model"], strict=True)
            self.refiner.to(self.device).eval()

    def predict(self, rgb: np.ndarray) -> Result:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected RGB uint8 image H x W x 3")
        padded, info = make_letterbox(rgb, self.image_size)
        x = (padded.astype(np.float32) / 255.0 - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None].to(self.device)
        with torch.inference_mode():
            outputs = self.model(x)
        # Move low-res results to CPU before restoring high-res; never upsample on GPU.
        names = ("final", "foreground", "boundary", "detail", "residual_background", "uncertainty", "objectness")
        maps = {}
        for key in names:
            logits = outputs[key].detach().float().sigmoid()[0, 0].cpu().numpy()
            maps[key] = unletterbox(logits, info)
        del outputs, x
        risk = risk_map(**maps)
        regions = select_rois(risk, patch=self.patch, overlap=self.overlap, max_rois=self.max_rois) if self.refiner else []
        probability = maps["final"]
        if self.refiner and regions:
            probability = weighted_refine(probability, risk, rgb, regions, self.refiner, self.device)
        alpha = binary_alpha(probability, self.threshold)
        return Result(rgb=rgb, alpha=alpha, probability=probability, risk=risk,
                      regions=regions, checkpoint_epoch=self.epoch, refiner_used=bool(self.refiner and regions))

    def run(self, source: str | Path, destination: str | Path,
            save_diagnostics: bool = False) -> Result:
        result = self.predict(read_rgb(source))
        result.save(destination, save_diagnostics=save_diagnostics)
        return result
