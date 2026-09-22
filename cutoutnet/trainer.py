from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from .dataset import PairedCutoutDataset
from .losses import CutoutNetLoss
from .metrics import binary_metrics, boundary_iou
from .model import CutoutNet


@dataclass
class TrainState:
    epoch: int = 0
    best_dice: float = -1.0


def _move_targets(targets: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in targets.items()}


def load_train_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(config_path: str = "config/cutoutnet_train.yaml") -> None:
    cfg = load_train_config(config_path)
    device_name = cfg.get("device", "auto")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    model_cfg = dict(cfg["model"])
    model = CutoutNet(**model_cfg).to(device)
    data_cfg = cfg["data"]
    train_ds = PairedCutoutDataset(
        data_cfg["train_root"],
        image_size=int(data_cfg["image_size"]),
        augment=True,
        target_cfg=data_cfg.get("targets", {}),
        seed=int(cfg.get("seed", 1337)),
    )
    val_ds = PairedCutoutDataset(
        data_cfg["val_root"],
        image_size=int(data_cfg["image_size"]),
        augment=False,
        target_cfg=data_cfg.get("targets", {}),
        seed=int(cfg.get("seed", 1337)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 4)),
        shuffle=True,
        num_workers=int(cfg["train"].get("workers", 2)),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(cfg["train"].get("val_batch_size", 2))),
        shuffle=False,
        num_workers=int(cfg["train"].get("workers", 2)),
        pin_memory=device.type == "cuda",
    )

    loss_fn = CutoutNetLoss(cfg.get("loss_weights"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 2e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    epochs = int(cfg["train"].get("epochs", 40))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    amp_enabled = device.type == "cuda" and bool(cfg["train"].get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    out_dir = Path(cfg["train"].get("output_dir", "checkpoints/cutoutnet"))
    out_dir.mkdir(parents=True, exist_ok=True)
    state = TrainState()
    history = []
    freeze_epochs = int(cfg["train"].get("freeze_encoder_epochs", 1))

    for epoch in range(1, epochs + 1):
        state.epoch = epoch
        model.freeze_encoder(epoch <= freeze_epochs)
        model.train()
        running = 0.0
        bar = tqdm(train_loader, desc=f"train {epoch}/{epochs}")
        for batch in bar:
            image = batch["image"].to(device, non_blocking=True)
            targets = _move_targets(batch["targets"], device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                pred = model(image)
                total, parts = loss_fn(pred, targets)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            running += float(total.detach().cpu())
            bar.set_postfix(loss=f"{float(total.detach().cpu()):.4f}")
        scheduler.step()

        model.eval()
        val_loss = 0.0
        metric_sums = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "boundary_iou": 0.0}
        batches = 0
        with torch.inference_mode():
            for batch in tqdm(val_loader, desc=f"val {epoch}/{epochs}"):
                image = batch["image"].to(device, non_blocking=True)
                targets = _move_targets(batch["targets"], device)
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                    pred = model(image)
                    total, _parts = loss_fn(pred, targets)
                val_loss += float(total.detach().cpu())
                m = binary_metrics(pred["final"], targets["final"])
                for k in ("iou", "dice", "precision", "recall"):
                    metric_sums[k] += m[k]
                metric_sums["boundary_iou"] += boundary_iou(pred["final"], targets["final"])
                batches += 1
        metrics = {k: v / max(1, batches) for k, v in metric_sums.items()}
        rec = {
            "epoch": epoch,
            "train_loss": running / max(1, len(train_loader)),
            "val_loss": val_loss / max(1, batches),
            **metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "time": time.time(),
        }
        history.append(rec)
        print(json.dumps(rec, indent=2))

        checkpoint = {
            "format": "cutoutnet-v0",
            "model": model.state_dict(),
            "model_config": model_cfg,
            "image_size": int(data_cfg["image_size"]),
            "epoch": epoch,
            "metrics": metrics,
        }
        torch.save(checkpoint, out_dir / "last.pt")
        if metrics["dice"] > state.best_dice:
            state.best_dice = metrics["dice"]
            torch.save(checkpoint, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Training complete. Best validation Dice: {state.best_dice:.5f}")
    print(f"Best checkpoint: {out_dir / 'best.pt'}")
