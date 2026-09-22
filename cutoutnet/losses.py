from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logits)
    dims = tuple(range(1, p.ndim))
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def focal_bce(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.5) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, p, 1.0 - p)
    at = torch.where(target > 0.5, torch.full_like(target, alpha), torch.full_like(target, 1.0 - alpha))
    return (at * (1.0 - pt).pow(gamma) * bce).mean()


class CutoutNetLoss(nn.Module):
    DEFAULT_WEIGHTS = {
        "final": 1.00,
        "foreground": 0.70,
        "boundary": 0.80,
        "detail": 0.65,
        "residual_background": 0.65,
        "uncertainty": 0.30,
        "objectness": 0.30,
    }

    def __init__(self, weights: dict | None = None):
        super().__init__()
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights:
            self.weights.update({k: float(v) for k, v in weights.items()})

    def forward(self, pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]):
        losses: dict[str, torch.Tensor] = {}
        losses["final"] = F.binary_cross_entropy_with_logits(pred["final"], target["final"]) + dice_loss(pred["final"], target["final"])
        losses["foreground"] = F.binary_cross_entropy_with_logits(pred["foreground"], target["foreground"]) + dice_loss(pred["foreground"], target["foreground"])
        losses["boundary"] = focal_bce(pred["boundary"], target["boundary"], gamma=2.0, alpha=0.70)
        losses["detail"] = focal_bce(pred["detail"], target["detail"], gamma=2.0, alpha=0.70)
        losses["residual_background"] = focal_bce(pred["residual_background"], target["residual_background"], gamma=2.0, alpha=0.65)
        losses["uncertainty"] = F.binary_cross_entropy_with_logits(pred["uncertainty"], target["uncertainty"])
        losses["objectness"] = F.binary_cross_entropy_with_logits(pred["objectness"], target["objectness"])
        total = sum(self.weights[k] * v for k, v in losses.items())
        return total, losses
