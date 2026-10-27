"""Sticker-focused optional supplemental losses for CutoutNet's seven logits.

Adds boundary-weighted segmentation and soft-clDice thin-structure preservation.
Enable explicitly: original v0.2 checkpoint parameters remain unchanged.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .losses import CutoutNetLoss


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    return torch.minimum(-F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0)),
                         -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1)))


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, 3, stride=1, padding=1)


def soft_skeleton(x: torch.Tensor, iterations: int = 8) -> torch.Tensor:
    """Differentiable approximation to 2D morphological skeleton."""
    x = x.float()
    opened = soft_dilate(soft_erode(x))
    skeleton = F.relu(x - opened)
    for _ in range(iterations):
        x = soft_erode(x)
        opened = soft_dilate(soft_erode(x))
        delta = F.relu(x - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def soft_cldice(logits: torch.Tensor, target: torch.Tensor, iterations: int = 8) -> torch.Tensor:
    """Topological penalty for thin structures, robust to empty-GT images."""
    p = logits.float().sigmoid()
    g = target.float()
    ps = soft_skeleton(p, iterations)
    gs = soft_skeleton(g, iterations)
    spatial = tuple(range(1, p.ndim))
    sensitivity = (ps * g).sum(spatial) / (ps.sum(spatial) + 1e-6)
    precision = (gs * p).sum(spatial) / (gs.sum(spatial) + 1e-6)
    cldice = 1 - (2 * sensitivity * precision + 1e-6) / (sensitivity + precision + 1e-6)
    valid = g.sum(spatial) > 0
    if not valid.any():
        return p.sum() * 0
    return cldice[valid].mean()


def weighted_edge_bce(logits: torch.Tensor, target: torch.Tensor, weight: float = 5.0) -> torch.Tensor:
    gt = target.float()
    boundary = (F.max_pool2d(gt, 5, stride=1, padding=2) -
                (-F.max_pool2d(-gt, 5, stride=1, padding=2))).clamp(0, 1)
    weights = 1 + float(weight) * boundary
    bce = F.binary_cross_entropy_with_logits(logits.float(), gt, reduction='none')
    return (weights * bce).sum() / weights.sum().clamp_min(1)


class StickerLoss(nn.Module):
    """Existing multi-head targets plus geometry-focused differentiable losses."""
    def __init__(self, weights: dict | None = None, *, edge_weight: float = .3,
                 topology_weight: float = .12, skeleton_iterations: int = 8):
        super().__init__()
        self.base = CutoutNetLoss(weights)
        self.edge_weight = float(edge_weight)
        self.topology_weight = float(topology_weight)
        self.iterations = skeleton_iterations

    def forward(self, preds: dict[str, torch.Tensor], target: dict[str, torch.Tensor]):
        total, parts = self.base(preds, target)
        parts = dict(parts)
        if self.edge_weight:
            parts['weighted_edge'] = weighted_edge_bce(preds['final'], target['final'])
            total = total + self.edge_weight * parts['weighted_edge']
        if self.topology_weight:
            parts['soft_cldice'] = soft_cldice(preds['final'], target['final'], self.iterations)
            total = total + self.topology_weight * parts['soft_cldice']
        return total, parts
