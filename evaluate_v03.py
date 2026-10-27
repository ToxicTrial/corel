"""Measure output quality at ORIGINAL image resolution, not training letterbox."""
from __future__ import annotations

import argparse
import json
import cv2
import numpy as np
from PIL import Image, ImageOps

from cutoutnet.dataset import PairedCutoutDataset
from cutoutnet.inference import CutoutPipeline, read_rgb


def _boundary(mask: np.ndarray, tol: int) -> np.ndarray:
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT,
                            np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)).astype(bool)


def measure(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    p, g = (pred >= 128), (true >= 128)
    intersection = int((p & g).sum())
    union = int((p | g).sum())
    tp, fp, fn = intersection, int((p & ~g).sum()), int((~p & g).sum())
    pb, gb = _boundary(p, 1), _boundary(g, 1)
    dilp = cv2.dilate(pb.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    dilg = cv2.dilate(gb.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    precision_b = (int((pb & dilg).sum()) + 1e-6) / (int(pb.sum()) + 1e-6)
    recall_b = (int((gb & dilp).sum()) + 1e-6) / (int(gb.sum()) + 1e-6)
    # Component recall is pixel overlap of each GT component with ANY predicted FG.
    n_comp, gt_labels = cv2.connectedComponents(g.astype(np.uint8), connectivity=8)
    retained = sum(bool((p & (gt_labels == j)).any()) for j in range(1, n_comp))
    # Count enclosed background via flood-fill from a padded edge.
    def holes(mask: np.ndarray) -> np.ndarray:
        inv = (~mask).astype(np.uint8)
        padded = np.pad(inv, 1)
        flood = padded.copy()
        ffmask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
        cv2.floodFill(flood, ffmask, (0, 0), 2)
        return (inv != 0) & (flood[1:-1, 1:-1] != 2)
    ph, gh = holes(p), holes(g)
    return {
        "iou": (tp + 1e-6) / (union + 1e-6),
        "dice": (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6),
        "precision": (tp + 1e-6) / (tp + fp + 1e-6),
        "recall": (tp + 1e-6) / (tp + fn + 1e-6),
        "boundary_f1": 2 * precision_b * recall_b / (precision_b + recall_b + 1e-6),
        "component_recall": retained / max(1, n_comp - 1),
        "hole_iou": (int((ph & gh).sum()) + 1e-6) / (int((ph | gh).sum()) + 1e-6),
        "residual_background_pixels": float(fp),
        "missing_foreground_pixels": float(fn),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate CutoutNet v0.3 on held-out paired test data")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default="data/cutoutnet/manifests/test.jsonl")
    p.add_argument("--refiner", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-rois", type=int, default=80)
    p.add_argument("--output", default="outputs/evaluation_v03.json")
    args = p.parse_args()
    ds = PairedCutoutDataset(args.data, image_size=64, augment=False)
    net = CutoutPipeline(args.checkpoint, refiner_checkpoint=args.refiner,
                         device=args.device, max_rois=args.max_rois)
    rows = []
    for img_path, mask_path in ds.pairs:
        rgb = read_rgb(img_path)
        with Image.open(mask_path) as im:
            mask = np.asarray(ImageOps.exif_transpose(im).convert("L"), dtype=np.uint8)
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"Image/mask size mismatch: {img_path} / {mask_path}")
        result = net.predict(rgb)
        row = {"image": str(img_path), **measure(result.alpha, mask)}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))
    cols = [k for k in rows[0] if k != "image"]
    summary = {k: float(np.mean([row[k] for row in rows])) for k in cols}
    from pathlib import Path
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"count": len(rows), "mean": summary, "images": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("AVERAGE:", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
