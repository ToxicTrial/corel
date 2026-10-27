"""Optional TRAINABLE instance segmentation (centers + center-offset vectors).

Unlike the binary foreground model, this module needs true *instance-ID* masks.
It never deletes predicted foreground: the reconstructed labels partition the
existing mask. Without trained instance weights, connected components remain
an explicitly acknowledged fallback (touching objects cannot be separated).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Sequential):
    def __init__(self, inc: int, outc: int):
        super().__init__(nn.Conv2d(inc, outc, 3, padding=1, bias=False),
                         nn.GroupNorm(min(8, outc), outc), nn.SiLU(inplace=True),
                         nn.Conv2d(outc, outc, 3, padding=1, bias=False),
                         nn.GroupNorm(min(8, outc), outc), nn.SiLU(inplace=True))


class InstanceNet(nn.Module):
    """4-channel RGB + foreground prior => center heatmap and normalized XY offsets."""
    def __init__(self, base: int = 24):
        super().__init__()
        self.base = base
        self.e1 = Block(4, base)
        self.e2 = Block(base, base * 2)
        self.e3 = Block(base * 2, base * 4)
        self.middle = Block(base * 4, base * 4)
        self.d2 = Block(base * 6, base * 2)
        self.d1 = Block(base * 3, base)
        self.out = nn.Conv2d(base, 3, 1)

    def forward(self, rgb: torch.Tensor, foreground: torch.Tensor) -> dict[str, torch.Tensor]:
        if rgb.shape[1] != 3 or foreground.shape[1] != 1:
            raise ValueError('Expected [N,3,H,W] RGB and [N,1,H,W] foreground')
        x = torch.cat((rgb, foreground), dim=1)
        a = self.e1(x)
        b = self.e2(F.avg_pool2d(a, 2, ceil_mode=True))
        c = self.e3(F.avg_pool2d(b, 2, ceil_mode=True))
        y = self.middle(c)
        y = self.d2(torch.cat((F.interpolate(y, size=b.shape[-2:], mode='bilinear', align_corners=False), b), dim=1))
        y = self.d1(torch.cat((F.interpolate(y, size=a.shape[-2:], mode='bilinear', align_corners=False), a), dim=1))
        z = self.out(y)
        return {'center': z[:, :1], 'offset': z[:, 1:].tanh()}


def make_instance_targets(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given IDs 0=background, 1..65535=objects, create supervised maps.

    Offset is normalized by image dimensions, targets in [-1,1].
    Empty objects and disconnected pieces belonging to one ID are supported.
    """
    ids = np.asarray(ids)
    if ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError('Instance mask must be a 2D integer ID map')
    h, w = ids.shape
    fg = ids > 0
    center = np.zeros((h, w), np.float32)
    offsets = np.zeros((2, h, w), np.float32)
    yy, xx = np.ogrid[:h, :w]
    for object_id in np.unique(ids[fg]):
        ys, xs = np.where(ids == object_id)
        cy, cx = float(ys.mean()), float(xs.mean())
        radius = max(2.0, min(18.0, .15 * np.sqrt(len(xs))))
        patch = np.exp(-((xx - cx)**2 + (yy - cy)**2) / (2.0 * radius * radius))
        center = np.maximum(center, patch.astype(np.float32))
        offsets[0, ys, xs] = (cx - xs) / max(1, w)
        offsets[1, ys, xs] = (cy - ys) / max(1, h)
    return center[None], offsets, fg.astype(np.float32)[None]


def instance_loss(output: dict[str, torch.Tensor], center: torch.Tensor,
                  offset: torch.Tensor, foreground: torch.Tensor) -> torch.Tensor:
    center_logits = output['center'].float()
    gt = center.float().clamp(0, 1)
    # Gaussian heatmap values below 1 are valid soft labels.
    heat = F.binary_cross_entropy_with_logits(center_logits, gt)
    weights = foreground.float()
    off = F.smooth_l1_loss(output['offset'].float(), offset.float(), reduction='none')
    return 5 * heat + 2 * (off * weights).sum() / (2 * weights.sum() + 1e-6)


def component_labels(mask: np.ndarray) -> np.ndarray:
    """Conservative fallback, preserves every foreground pixel, even a single dot."""
    _, ids = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
    return ids.astype(np.int32)


def decode_instances(mask: np.ndarray, centers: np.ndarray, offsets: np.ndarray,
                     peak_threshold: float = .28, min_peak_distance: int = 8) -> np.ndarray:
    """Split foreground components using predicted centers and per-pixel votes.

    Every positive pixel receives exactly one positive ID, background stays 0.
    Conservative fallback avoids deleting objects if the instance CNN is unsure.
    """
    fg = np.asarray(mask) > 0
    if fg.ndim != 2 or centers.shape != fg.shape or offsets.shape != (2, *fg.shape):
        raise ValueError('Mask, centers and offset resolutions must match')
    h, w = fg.shape
    result = np.zeros((h, w), np.int32)
    if not fg.any():
        return result
    comps = component_labels(fg)
    scores = np.clip(centers.astype(np.float32), 0, 1)
    size = max(3, min_peak_distance * 2 + 1)
    if size % 2 == 0:
        size += 1
    is_max = scores >= cv2.dilate(scores, np.ones((size, size), np.uint8)) - 1e-7
    peak_map = ((scores >= peak_threshold) & is_max & fg).astype(np.uint8)
    _, peak_labs, _, peak_centroids = cv2.connectedComponentsWithStats(peak_map, connectivity=8)
    peaks = peak_centroids[1:]  # x, y
    k = 1
    for compid in np.unique(comps[fg]):
        ys, xs = np.where(comps == compid)
        if len(xs) == 0:
            continue
        local = peaks[comps[np.clip(np.round(peaks[:, 1]).astype(int), 0, h-1),
                            np.clip(np.round(peaks[:, 0]).astype(int), 0, w-1)] == compid]
        if len(local) < 2:
            result[ys, xs] = k
            k += 1
            continue
        votes = np.stack((xs + offsets[0, ys, xs] * w,
                          ys + offsets[1, ys, xs] * h), axis=1)
        # Chunk to keep memory bounded with hundreds of thousands of pixels.
        for begin in range(0, len(xs), 4096):
            end = min(len(xs), begin + 4096)
            d = ((votes[begin:end, None, :] - local[None, :, :]) ** 2).sum(axis=2)
            result[ys[begin:end], xs[begin:end]] = k + d.argmin(axis=1)
        k += len(local)
    return result


def infer_instance_labels(model: InstanceNet, rgb: np.ndarray, probability: np.ndarray,
                          *, device: torch.device, side: int = 640, threshold: float = .5) -> np.ndarray:
    """Perform instance CNN inference at working resolution, lift labels to original.

    Uses nearest-neighbor label resizing followed by conservative reconstruction
    so no foreground pixels disappear from the final union.
    """
    h, w = probability.shape
    if rgb.shape[:2] != (h, w):
        raise ValueError('RGB/mask size mismatch')
    scale = min(side / h, side / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    input_rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    input_fg = cv2.resize(probability, (nw, nh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    x = torch.from_numpy(np.ascontiguousarray(input_rgb.transpose(2, 0, 1)))[None].to(device)
    p = torch.from_numpy(input_fg[None, None].copy()).to(device)
    model.eval()
    with torch.inference_mode():
        pred = model(x, p)
    centers = pred['center'].sigmoid()[0, 0].cpu().numpy()
    offsets = pred['offset'][0].cpu().numpy()
    fg = input_fg >= threshold
    low_ids = decode_instances(fg, centers, offsets)
    ids = cv2.resize(low_ids.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
    original_fg = probability >= threshold
    ids[~original_fg] = 0
    if np.any(original_fg & (ids == 0)):
        # Find the closest existing foreground instance for missing fine pixels.
        missing = original_fg & (ids == 0)
        if (ids > 0).any():
            _, inds = cv2.distanceTransformWithLabels((ids == 0).astype(np.uint8),
                         cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
            # OpenCV labels are ordinal zero-pixel indices, not instance IDs.
            _, zeros_idx = np.unique(inds[ids > 0], return_index=True)
            nonzero_labels = inds[ids > 0]
            orig_ids = ids[ids > 0]
            lut = np.zeros(int(inds.max()) + 1, dtype=np.int32)
            lut[nonzero_labels] = orig_ids
            ids[missing] = lut[inds[missing]]
        if np.any(original_fg & (ids == 0)):
            cc = component_labels(original_fg & (ids == 0))
            cc[cc > 0] += int(ids.max())
            ids[cc > 0] = cc[cc > 0]
    return ids


def masks_by_id(ids: np.ndarray) -> dict[int, np.ndarray]:
    return {int(i): (ids == i).astype(np.uint8) * 255 for i in np.unique(ids) if i > 0}


def load_instance_net(path: str | Path, device: torch.device) -> InstanceNet:
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt.get('format') != 'cutoutnet-instance-v1':
        raise ValueError('Wrong instance checkpoint format')
    model = InstanceNet(base=int(ckpt['base']))
    model.load_state_dict(ckpt['model'])
    return model.to(device).eval()
