from __future__ import annotations

import cv2
import numpy as np


def _binary(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 127).astype(np.uint8)


def boundary_target(mask: np.ndarray, width: int = 2) -> np.ndarray:
    m = _binary(mask)
    k = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(m, k, iterations=max(1, int(width)))
    ero = cv2.erode(m, k, iterations=max(1, int(width)))
    return (dil != ero).astype(np.float32)


def detail_target(mask: np.ndarray, width: int = 4) -> np.ndarray:
    """Foreground pixels close to a contour + very thin foreground structures."""
    m = _binary(mask)
    inside = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    band = ((inside > 0) & (inside <= width)) | ((outside > 0) & (outside <= width))
    return band.astype(np.float32)


def residual_background_target(mask: np.ndarray, width: int = 12) -> np.ndarray:
    """Hard background: cavities and pixels close enough to be confused with FG."""
    m = _binary(mask)
    dist = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
    hard = (m == 0) & (dist <= max(1, int(width)))

    # Explicitly emphasize enclosed holes. Flood-fill background from canvas edge;
    # remaining background islands are true internal cavities.
    inv = (1 - m).astype(np.uint8)
    h, w = inv.shape
    flood = inv.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 2)
    holes = inv & (flood != 2)
    return (hard | holes.astype(bool)).astype(np.float32)


def uncertainty_target(mask: np.ndarray, width: int = 6) -> np.ndarray:
    return detail_target(mask, width=width)


def objectness_target(mask: np.ndarray) -> np.ndarray:
    """Per-object center support map derived from normalized distance transforms.

    Every connected foreground component gets its own 0..1 center heatmap so a
    small sixth sticker can contribute independently from the largest object.
    """
    m = _binary(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros(m.shape, np.float32)
    for label in range(1, n):
        comp = (labels == label).astype(np.uint8)
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        d = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        mx = float(d.max())
        if mx > 0:
            out = np.maximum(out, d / mx)
    return out


def build_targets(mask: np.ndarray, cfg: dict | None = None) -> dict[str, np.ndarray]:
    cfg = cfg or {}
    m = _binary(mask).astype(np.float32)
    return {
        "final": m,
        "foreground": m,
        "boundary": boundary_target(mask, int(cfg.get("boundary_width", 2))),
        "detail": detail_target(mask, int(cfg.get("detail_width", 4))),
        "residual_background": residual_background_target(mask, int(cfg.get("residual_bg_width", 12))),
        "uncertainty": uncertainty_target(mask, int(cfg.get("uncertainty_width", 6))),
        "objectness": objectness_target(mask),
    }
