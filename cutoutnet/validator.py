"""Prioritize risky, high-resolution regions without deleting any objects.

Validator generates *requests* for refinement. The final segmentation is never
silently changed by connected-component or hole heuristics.
"""
from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class ROI:
    x0: int
    y0: int
    x1: int
    y1: int
    score: float

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x1, self.y1


def _prob(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, np.float32), 0.0, 1.0)


def risk_map(final: np.ndarray, foreground: np.ndarray | None = None,
             boundary: np.ndarray | None = None, detail: np.ndarray | None = None,
             residual_background: np.ndarray | None = None,
             uncertainty: np.ndarray | None = None,
             objectness: np.ndarray | None = None) -> np.ndarray:
    """Rank pixels needing extra work. No score implies a pixel may be removed.

    All maps must already be in original image coordinates.
    """
    p = _prob(final)
    hard = (p >= 0.5).astype(np.uint8)
    # Dilate/erode with 3x3, preserving boundary visibility at the image edge.
    edge = cv2.morphologyEx(hard, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    edge = cv2.dilate(edge, np.ones((9, 9), np.uint8)).astype(np.float32)
    ambiguity = (1.0 - 2.0 * np.abs(p - 0.5)).clip(0, 1)
    risk = 0.30 * edge + 0.25 * ambiguity
    if foreground is not None:
        risk += 0.20 * np.abs(_prob(foreground) - p)
    if boundary is not None:
        risk += 0.12 * _prob(boundary)
    if detail is not None:
        risk += 0.10 * _prob(detail)
    if residual_background is not None:
        # Background suspected close to an existing object.
        nearby = cv2.dilate(hard, np.ones((13, 13), np.uint8)).astype(np.float32)
        risk += 0.13 * nearby * _prob(residual_background)
    if uncertainty is not None:
        risk += 0.15 * _prob(uncertainty)
    if objectness is not None:
        risk += 0.05 * _prob(objectness) * ambiguity
    return np.clip(risk, 0, 1).astype(np.float32)


def select_rois(risk: np.ndarray, patch: int = 512, overlap: int = 128,
                max_rois: int = 80, minimum_score: float = 0.01) -> list[ROI]:
    """Select high-risk tiles in deterministic ranked order.

    Strided tiles cover every pixel; no component, including tiny isolated
    details, is filtered by area. `max_rois=0` means no refinement.
    """
    if max_rois < 1:
        return []
    h, w = risk.shape
    if patch < 16 or overlap < 0 or overlap >= patch:
        raise ValueError("patch must be >=16 and 0 <= overlap < patch")

    def starts(length: int) -> list[int]:
        if length <= patch:
            return [0]
        last = length - patch
        return sorted(set([*range(0, last, patch - overlap), last]))

    ys, xs = starts(h), starts(w)
    # Box scores from integral image: O(1) after linear preprocessing.
    ii = cv2.integral(risk.astype(np.float64))
    candidates: list[ROI] = []
    for y0 in ys:
        for x0 in xs:
            x1, y1 = min(x0 + patch, w), min(y0 + patch, h)
            summ = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
            score = float(summ / max(1, (x1 - x0) * (y1 - y0)))
            if score >= minimum_score:
                candidates.append(ROI(x0, y0, x1, y1, score))
    candidates.sort(key=lambda r: (-r.score, r.y0, r.x0))
    return candidates[:max_rois]


def binary_alpha(probability: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    p = _prob(probability)
    return np.where(p >= threshold, 255, 0).astype(np.uint8)
