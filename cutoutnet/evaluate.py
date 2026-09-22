from __future__ import annotations

from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import PairedCutoutDataset
from .metrics import binary_metrics, boundary_iou
from .model import CutoutNet


def evaluate(checkpoint: str, data_root: str, image_size: int = 768, batch_size: int = 2, device: str = "auto") -> dict:
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ckpt["model_config"])
    cfg["pretrained_backbone"] = False
    model = CutoutNet(**cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(dev)
    ds = PairedCutoutDataset(data_root, image_size=image_size, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    sums = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "boundary_iou": 0.0}
    n = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="evaluate"):
            image = batch["image"].to(dev)
            gt = batch["targets"]["final"].to(dev)
            pred = model(image)["final"]
            m = binary_metrics(pred, gt)
            for k in ("iou", "dice", "precision", "recall"):
                sums[k] += m[k]
            sums["boundary_iou"] += boundary_iou(pred, gt)
            n += 1
    result = {k: v / max(1, n) for k, v in sums.items()}
    print(json.dumps(result, indent=2))
    return result
