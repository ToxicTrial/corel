from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import random
import shutil
import time

import numpy as np
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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_text_write(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _resolve_resume(cfg: dict, out_dir: Path) -> Path | None:
    value = cfg["train"].get("resume", "auto")
    if value in (None, False, "", "none", "None"):
        return None
    if str(value).lower() == "auto":
        p = out_dir / "last_full.pt"
        return p if p.exists() else None
    p = Path(str(value)).expanduser()
    return p if p.exists() else None


def train(config_path: str = "config/train.yaml") -> None:
    cfg = load_train_config(config_path)
    seed = int(cfg.get("seed", 1337))
    _set_seed(seed)

    device_name = cfg.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    model_cfg = dict(cfg["model"])
    model = CutoutNet(**model_cfg).to(device)
    data_cfg = cfg["data"]
    train_ds = PairedCutoutDataset(
        data_cfg["train_root"],
        image_size=int(data_cfg["image_size"]),
        augment=True,
        target_cfg=data_cfg.get("targets", {}),
        seed=seed,
    )
    val_ds = PairedCutoutDataset(
        data_cfg["val_root"],
        image_size=int(data_cfg["image_size"]),
        augment=False,
        target_cfg=data_cfg.get("targets", {}),
        seed=seed,
    )

    workers = int(cfg["train"].get("workers", 2))
    loader_kwargs = dict(num_workers=workers, pin_memory=device.type == "cuda")
    if workers > 0:
        loader_kwargs["persistent_workers"] = bool(cfg["train"].get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 2))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 2)),
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(cfg["train"].get("val_batch_size", 1))),
        shuffle=False,
        **loader_kwargs,
    )

    loss_fn = CutoutNetLoss(cfg.get("loss_weights"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 2e-4)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    epochs = int(cfg["train"].get("epochs", 40))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=float(cfg["train"].get("min_lr", 1e-6)))
    amp_enabled = device.type == "cuda" and bool(cfg["train"].get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    out_dir = Path(cfg["train"].get("output_dir", "checkpoints/cutoutnet")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    state = TrainState()
    history: list[dict] = []
    start_epoch = 1

    resume_path = _resolve_resume(cfg, out_dir)
    if resume_path is not None:
        print(f"Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        state.best_dice = float(ckpt.get("best_dice", ckpt.get("metrics", {}).get("dice", -1.0)))
        history = list(ckpt.get("history", []))
        print(f"Resume epoch={start_epoch}, best_dice={state.best_dice:.5f}")

    if start_epoch > epochs:
        print(f"Checkpoint is already at epoch {start_epoch-1}; target epochs={epochs}. Nothing to do.")
        return

    freeze_epochs = int(cfg["train"].get("freeze_encoder_epochs", 1))
    accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    save_every = max(1, int(cfg["train"].get("save_every_epochs", 1)))
    val_every = max(1, int(cfg["train"].get("val_every_epochs", 1)))
    grad_clip = float(cfg["train"].get("grad_clip", 1.0))

    print(f"Training samples: {len(train_ds)} | validation: {len(val_ds)}")
    print(f"image_size={data_cfg['image_size']} batch={cfg['train'].get('batch_size', 2)} accum={accum_steps} effective_batch={int(cfg['train'].get('batch_size',2))*accum_steps}")

    for epoch in range(start_epoch, epochs + 1):
        state.epoch = epoch
        model.freeze_encoder(epoch <= freeze_epochs)
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        bar = tqdm(train_loader, desc=f"train {epoch}/{epochs}")
        for step, batch in enumerate(bar, 1):
            image = batch["image"].to(device, non_blocking=True)
            targets = _move_targets(batch["targets"], device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                pred = model(image)
                total, _parts = loss_fn(pred, targets)
                scaled_total = total / accum_steps
            scaler.scale(scaled_total).backward()

            do_step = step % accum_steps == 0 or step == len(train_loader)
            if do_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            loss_value = float(total.detach().cpu())
            running += loss_value
            if device.type == "cuda":
                mem = torch.cuda.max_memory_allocated() / 1024**3
                bar.set_postfix(loss=f"{loss_value:.4f}", vram=f"{mem:.1f}G")
            else:
                bar.set_postfix(loss=f"{loss_value:.4f}")

        scheduler.step()

        metrics = {"iou": float("nan"), "dice": float("nan"), "precision": float("nan"), "recall": float("nan"), "boundary_iou": float("nan")}
        val_loss = float("nan")
        if epoch % val_every == 0 or epoch == epochs:
            model.eval()
            val_loss_sum = 0.0
            metric_sums = {"iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0, "boundary_iou": 0.0}
            batches = 0
            with torch.inference_mode():
                for batch in tqdm(val_loader, desc=f"val {epoch}/{epochs}"):
                    image = batch["image"].to(device, non_blocking=True)
                    targets = _move_targets(batch["targets"], device)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                        pred = model(image)
                        total, _parts = loss_fn(pred, targets)
                    val_loss_sum += float(total.detach().cpu())
                    m = binary_metrics(pred["final"], targets["final"])
                    for k in ("iou", "dice", "precision", "recall"):
                        metric_sums[k] += m[k]
                    metric_sums["boundary_iou"] += boundary_iou(pred["final"], targets["final"])
                    batches += 1
            metrics = {k: v / max(1, batches) for k, v in metric_sums.items()}
            val_loss = val_loss_sum / max(1, batches)

        rec = {
            "epoch": epoch,
            "train_loss": running / max(1, len(train_loader)),
            "val_loss": val_loss,
            **metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "time": time.time(),
        }
        history.append(rec)
        print(json.dumps(rec, indent=2))

        if not np.isnan(metrics["dice"]) and metrics["dice"] > state.best_dice:
            state.best_dice = metrics["dice"]
            best = {
                "format": "cutoutnet-v0",
                "model": model.state_dict(),
                "model_config": model_cfg,
                "image_size": int(data_cfg["image_size"]),
                "epoch": epoch,
                "metrics": metrics,
            }
            _atomic_torch_save(best, out_dir / "best.pt")

        full_checkpoint = {
            "format": "cutoutnet-train-v1",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if amp_enabled else None,
            "model_config": model_cfg,
            "image_size": int(data_cfg["image_size"]),
            "epoch": epoch,
            "best_dice": state.best_dice,
            "metrics": metrics,
            "history": history,
            "config": cfg,
        }
        if epoch % save_every == 0 or epoch == epochs:
            _atomic_torch_save(full_checkpoint, out_dir / "last_full.pt")
            # Lightweight inference last checkpoint as well.
            last = {
                "format": "cutoutnet-v0",
                "model": model.state_dict(),
                "model_config": model_cfg,
                "image_size": int(data_cfg["image_size"]),
                "epoch": epoch,
                "metrics": metrics,
            }
            _atomic_torch_save(last, out_dir / "last.pt")
        _atomic_text_write(json.dumps(history, indent=2), out_dir / "history.json")

    print(f"Training complete. Best validation Dice: {state.best_dice:.5f}")
    print(f"Best checkpoint: {out_dir / 'best.pt'}")
    print(f"Resume checkpoint: {out_dir / 'last_full.pt'}")
