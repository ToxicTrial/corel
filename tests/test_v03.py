import numpy as np
import cv2
from pathlib import Path

from cutoutnet.validator import risk_map, select_rois, binary_alpha
from cutoutnet.sticker import cut_contours, save_cut_svg
from cutoutnet.inference import make_letterbox, unletterbox, weighted_refine
from evaluate_v03 import measure


def test_preserves_multiple_components_and_hole():
    gt = np.zeros((120, 160), dtype=np.uint8)
    cv2.circle(gt, (32, 50), 24, 255, -1)
    cv2.circle(gt, (32, 50), 11, 0, -1)
    cv2.rectangle(gt, (100, 20), (140, 105), 255, -1)
    prob = gt.astype(np.float32) / 255
    out = binary_alpha(prob)
    assert np.array_equal(gt, out)
    assert out[50, 32] == 0  # genuine internal hole
    assert out[60, 118] == 255  # independent second object
    roi = select_rois(risk_map(prob), patch=64, overlap=16, max_rois=40)
    assert roi
    assert any(r.x0 <= 32 < r.x1 for r in roi)
    assert any(r.x0 <= 118 < r.x1 for r in roi)


def test_sticker_vector_export(tmp_path):
    mask = np.zeros((128, 160), dtype=np.uint8)
    cv2.circle(mask, (20, 20), 10, 255, -1)
    cv2.circle(mask, (80, 70), 15, 255, -1)
    cv2.circle(mask, (80, 70), 5, 0, -1)
    contours = cut_contours(mask, offset_mm=2, dpi=300)
    assert len(contours) == 2
    path = tmp_path / "cut.svg"
    assert save_cut_svg(mask, path) == 2
    assert '<svg xmlns="http://www.w3.org/2000/svg"' in path.read_text()
    assert 'mm"' in path.read_text()
    assert save_cut_svg(mask, tmp_path / "unified.svg", mode="unified") == 1


def test_letterbox_exact_original_size():
    rgb = np.ones((100, 200, 3), np.uint8) * 127
    img, info = make_letterbox(rgb, 64)
    assert img.shape == (64, 64, 3)
    unboxed = unletterbox(np.ones((64, 64), np.float32), info)
    assert unboxed.shape == (100, 200)
    assert np.allclose(unboxed, 1)


def test_metrics_perfect_and_holes():
    mask = np.zeros((64, 64), np.uint8)
    cv2.rectangle(mask, (10, 10), (50, 50), 255, -1)
    cv2.circle(mask, (25, 25), 5, 0, -1)
    m = measure(mask, mask)
    assert m["dice"] > .9999
    assert m["hole_iou"] > .9999
    assert m["component_recall"] == 1


def test_bad_threshold_rejected():
    import pytest
    with pytest.raises(ValueError):
        binary_alpha(np.zeros((1, 2)), threshold=2)


def test_tiny_dot_has_a_cutline():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[30, 30] = 255
    lines = cut_contours(mask, offset_mm=0, dpi=300)
    assert len(lines) == 1
    assert len(lines[0]) >= 3


def test_roi_identity_when_zero_correction():
    import torch
    from cutoutnet.validator import ROI

    class ZeroRefiner(torch.nn.Module):
        def forward(self, rgb, coarse, uncertainty):
            return torch.zeros_like(coarse)

    prob = np.random.default_rng(42).uniform(.1, .9, (65, 75)).astype(np.float32)
    rgb = np.zeros((65, 75, 3), np.uint8)
    unc = np.full_like(prob, .5)
    rois = [ROI(0, 0, 50, 50, .9), ROI(25, 15, 75, 65, .8)]
    result = weighted_refine(prob, unc, rgb, rois, ZeroRefiner(), torch.device("cpu"))
    assert np.allclose(result, prob, atol=1e-6)
