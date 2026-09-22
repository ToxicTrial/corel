from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    pred = torch.sigmoid(logits) >= threshold
    gt = target >= 0.5
    dims = tuple(range(1, pred.ndim))
    tp = (pred & gt).sum(dims).float()
    fp = (pred & ~gt).sum(dims).float()
    fn = (~pred & gt).sum(dims).float()
    eps = 1e-6
    iou = ((tp + eps) / (tp + fp + fn + eps)).mean().item()
    dice = ((2 * tp + eps) / (2 * tp + fp + fn + eps)).mean().item()
    precision = ((tp + eps) / (tp + fp + eps)).mean().item()
    recall = ((tp + eps) / (tp + fn + eps)).mean().item()
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}


def boundary_iou(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5, width: int = 2) -> float:
    pred = (torch.sigmoid(logits) >= threshold).float()
    gt = (target >= 0.5).float()
    k = 2 * width + 1
    p_dil = F.max_pool2d(pred, k, stride=1, padding=width)
    p_ero = -F.max_pool2d(-pred, k, stride=1, padding=width)
    g_dil = F.max_pool2d(gt, k, stride=1, padding=width)
    g_ero = -F.max_pool2d(-gt, k, stride=1, padding=width)
    pb = (p_dil - p_ero) > 0
    gb = (g_dil - g_ero) > 0
    inter = (pb & gb).sum().float()
    union = (pb | gb).sum().float()
    return float(((inter + 1e-6) / (union + 1e-6)).item())
