from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _fp32_pair(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute numerically sensitive losses in FP32 even when model forward uses AMP/FP16."""
    return logits.float(), target.float()


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    logits, target = _fp32_pair(logits, target)
    p = torch.sigmoid(logits)
    dims = tuple(range(1, p.ndim))
    inter = (p * target).sum(dims)
    denom = p.sum(dims) + target.sum(dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def focal_bce(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.5) -> torch.Tensor:
    # Keep this entirely in FP32. Under autocast, FP16 saturation in sigmoid/focal
    # weighting can otherwise create 0 * inf -> NaN on very confident logits.
    logits, target = _fp32_pair(logits, target)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(target > 0.5, p, 1.0 - p)
    at = torch.where(
        target > 0.5,
        torch.full_like(target, alpha),
        torch.full_like(target, 1.0 - alpha),
    )
    focal = (1.0 - pt).clamp_(0.0, 1.0).pow(gamma)
    return (at * focal * bce).mean()


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
        # Explicit FP32 loss math is intentional: the network can still run under
        # autocast/FP16, while BCE/focal/dice remain stable.
        p = {k: v.float() for k, v in pred.items()}
        t = {k: v.float() for k, v in target.items()}

        losses: dict[str, torch.Tensor] = {}
        losses["final"] = F.binary_cross_entropy_with_logits(p["final"], t["final"]) + dice_loss(p["final"], t["final"])
        losses["foreground"] = F.binary_cross_entropy_with_logits(p["foreground"], t["foreground"]) + dice_loss(p["foreground"], t["foreground"])
        losses["boundary"] = focal_bce(p["boundary"], t["boundary"], gamma=2.0, alpha=0.70)
        losses["detail"] = focal_bce(p["detail"], t["detail"], gamma=2.0, alpha=0.70)
        losses["residual_background"] = focal_bce(p["residual_background"], t["residual_background"], gamma=2.0, alpha=0.65)
        losses["uncertainty"] = F.binary_cross_entropy_with_logits(p["uncertainty"], t["uncertainty"])
        losses["objectness"] = F.binary_cross_entropy_with_logits(p["objectness"], t["objectness"])

        total = sum(self.weights[k] * v for k, v in losses.items())
        return total, losses
