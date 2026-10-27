"""Optional learned RGB edge decontamination for strictly binary-alpha stickers.

Requires its own supervised training on CLEAN RGBA cutouts. Without a trained
checkpoint, returns input RGB unchanged. Alpha is never modified.
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ColorCleanNet(nn.Module):
    """Residual color correction using RGB and the final binary segmentation."""
    def __init__(self, base: int = 24):
        super().__init__()
        self.base = base
        self.net = nn.Sequential(
            nn.Conv2d(4, base, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=2, dilation=2), nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=4, dilation=4), nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base, 3, 1), nn.Tanh(),
        )

    def forward(self, rgb: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat((rgb, alpha), 1)) * .45
        return (rgb + delta).clamp(0, 1)


def edge_band(alpha: np.ndarray, radius: int = 5) -> np.ndarray:
    if radius < 1:
        return np.zeros_like(alpha, np.bool_)
    m = (alpha > 127).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return ((m > 0) & (cv2.erode(m, k) == 0))


def apply_color_net(rgb: np.ndarray, alpha: np.ndarray, model: ColorCleanNet,
                    device: torch.device, patch: int = 512, margin: int = 24,
                    band: int = 5) -> np.ndarray:
    """Color-clean only foreground pixels near alpha boundary, never alter alpha.

    Tiles overlap in context, but only their central core is committed, avoiding
    seams. The network is strictly optional and must be trained separately.
    """
    if rgb.shape[:2] != alpha.shape or rgb.shape[2] != 3:
        raise ValueError('RGB/alpha size mismatch')
    if margin >= patch // 2 or patch < 32:
        raise ValueError('patch must be >2*margin and >=32')
    h, w = alpha.shape
    band_map = edge_band(alpha, band)
    result = rgb.copy()
    if not band_map.any():
        return result
    model.eval()
    # Each non-overlapping core has extra context margin for convolution.
    step = patch - 2 * margin
    with torch.inference_mode():
        for y in range(0, h, step):
            for x in range(0, w, step):
                cy1, cx1 = min(y + step, h), min(x + step, w)
                if not band_map[y:cy1, x:cx1].any():
                    continue
                y0, x0 = max(0, y - margin), max(0, x - margin)
                y1, x1 = min(h, cy1 + margin), min(w, cx1 + margin)
                a = torch.from_numpy((alpha[y0:y1, x0:x1] > 127).astype(np.float32)[None, None]).to(device)
                src = torch.from_numpy(np.ascontiguousarray(rgb[y0:y1, x0:x1].transpose(2, 0, 1)))[None].float().to(device) / 255
                clean = model(src, a)[0].permute(1, 2, 0).cpu().numpy()
                cleaned = np.clip(np.rint(clean * 255), 0, 255).astype(np.uint8)
                core_mask = band_map[y:cy1, x:cx1]
                target = result[y:cy1, x:cx1]
                candidate = cleaned[y-y0:cy1-y0, x-x0:cx1-x0]
                target[core_mask] = candidate[core_mask]
    return result


def load_color_net(path: str | Path, device: torch.device) -> ColorCleanNet:
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt.get('format') != 'cutoutnet-color-v1':
        raise ValueError('Wrong color checkpoint format')
    model = ColorCleanNet(base=int(ckpt['base']))
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval()
