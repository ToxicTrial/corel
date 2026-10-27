"""Physical sticker cut-line export independent of binary alpha segmentation."""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape
import cv2
import numpy as np


def cut_contours(alpha: np.ndarray, offset_mm: float, dpi: float,
                 mode: str = "separate", simplify_px: float = 1.0) -> list[np.ndarray]:
    if mode not in {"separate", "unified"}:
        raise ValueError("mode must be separate or unified")
    if dpi <= 0 or offset_mm < 0:
        raise ValueError("dpi must be >0 and offset_mm >=0")
    mask = (alpha >= 128).astype(np.uint8)
    pad = int(np.ceil(offset_mm * dpi / 25.4)) + 3
    # Prevent clipping stickers touching source image edges.
    padded = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    r = int(round(offset_mm * dpi / 25.4))
    if r > 0:
        size = 2 * r + 1
        padded = cv2.dilate(padded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # One-pixel dots and 1px wires are valid printed content too. OpenCV gives
    # a zero-area contour for them; wrap them in an outer pixel-sized rectangle.
    repaired = []
    for c in contours:
        if cv2.contourArea(c) > 0:
            repaired.append(c)
        else:
            x, y, ww, hh = cv2.boundingRect(c)
            repaired.append(np.array([[[x, y]], [[x + ww, y]],
                                      [[x + ww, y + hh]], [[x, y + hh]]], dtype=np.int32))
    contours = repaired
    if mode == "unified" and len(contours) > 1:
        # Conservative one-piece outer cut; can be larger than a tight union.
        contours = [cv2.convexHull(np.concatenate(contours, axis=0))]
    output = []
    for c in contours:
        if simplify_px > 0:
            reduced = cv2.approxPolyDP(c, simplify_px, True)
            if len(reduced) >= 3:
                c = reduced
        c = c.reshape(-1, 2).astype(np.float32)
        c[:, 0] -= pad
        c[:, 1] -= pad
        if len(c) >= 3:
            output.append(c)
    return sorted(output, key=lambda c: -abs(cv2.contourArea(c)))


def save_cut_svg(alpha: np.ndarray, destination: str | Path, *,
                 offset_mm: float = 2.0, dpi: float = 300.0,
                 mode: str = "separate", simplify_px: float = 1.0) -> int:
    h, w = alpha.shape
    contours = cut_contours(alpha, offset_mm, dpi, mode, simplify_px)
    if not contours:
        raise ValueError("Empty foreground mask: no contour to export")
    margin = int(np.ceil(offset_mm * dpi / 25.4)) + 4
    width_px, height_px = w + 2 * margin, h + 2 * margin
    width_mm, height_mm = width_px * 25.4 / dpi, height_px * 25.4 / dpi
    paths = []
    for points in contours:
        points += margin
        coords = [f"{float(x):.2f},{float(y):.2f}" for x, y in points]
        paths.append(f'<path d="M {escape(" L ".join(coords))} Z" fill="none" stroke="red" stroke-width="0.5"/>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width_mm:.4f}mm" '
           f'height="{height_mm:.4f}mm" viewBox="0 0 {width_px} {height_px}">\n'
           + "\n".join(paths) + "\n</svg>\n")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(svg, encoding="utf-8")
    return len(contours)
