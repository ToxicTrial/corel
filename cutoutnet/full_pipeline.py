"""CutoutNet v0.4: single interface to ALL trained, optional CNN components.

Base model is mandatory. Instance separation, ROI refinement and edge-color
cleanup are independent trainable modules activated only with valid checkpoints.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch

from .inference import CutoutPipeline, Result, read_rgb
from .instances import load_instance_net, infer_instance_labels, component_labels, masks_by_id
from .colors import load_color_net, apply_color_net


@dataclass
class FullResult:
    base: Result
    rgb: np.ndarray
    alpha: np.ndarray
    instance_ids: np.ndarray
    selected_ids: list[int]
    instance_trained: bool
    color_trained: bool

    def rgba(self) -> np.ndarray:
        rgba = np.dstack((self.rgb.copy(), self.alpha))
        rgba[self.alpha == 0, :3] = 0
        return rgba

    def save(self, path: str | Path, *, diagnostics: bool = False,
             individual: bool = False) -> None:
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.rgba(), 'RGBA').save(dest)
        if individual:
            for object_id in self.selected_ids:
                alpha = (self.instance_ids == object_id).astype(np.uint8) * 255
                rgba = np.dstack((self.rgb.copy(), alpha))
                rgba[alpha == 0, :3] = 0
                Image.fromarray(rgba, 'RGBA').save(dest.with_name(f'{dest.stem}_object_{object_id:03}.png'))
        if diagnostics:
            import json
            Image.fromarray(self.alpha).save(dest.with_name(dest.stem + '_mask.png'))
            Image.fromarray((self.base.risk.clip(0, 1) * 255).astype('uint8')).save(dest.with_name(dest.stem + '_risk.png'))
            stats = {'instance_ids': self.selected_ids, 'instance_count': len(self.selected_ids),
                     'instance_model_used': self.instance_trained, 'color_model_used': self.color_trained,
                     'refiner_used': self.base.refiner_used,
                     'checkpoint_epoch': self.base.checkpoint_epoch,
                     'rois': [list(x.box) for x in self.base.regions]}
            dest.with_name(dest.stem + '_diagnostics.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')


class FullPipeline:
    def __init__(self, checkpoint: str | Path, *, refiner_checkpoint: str | Path | None = None,
                 instance_checkpoint: str | Path | None = None,
                 color_checkpoint: str | Path | None = None,
                 device: str = 'auto', image_size: int | None = None,
                 threshold: float = .5, patch: int = 512, overlap: int = 128,
                 max_rois: int = 80, instance_side: int = 640):
        self.base = CutoutPipeline(checkpoint, refiner_checkpoint=refiner_checkpoint,
                                   device=device, image_size=image_size, threshold=threshold,
                                   patch=patch, overlap=overlap, max_rois=max_rois)
        self.instance = load_instance_net(instance_checkpoint, self.base.device) if instance_checkpoint else None
        self.color = load_color_net(color_checkpoint, self.base.device) if color_checkpoint else None
        self.side = instance_side
        self.threshold = threshold

    def predict(self, rgb: np.ndarray, *, keep_ids: list[int] | None = None) -> FullResult:
        base = self.base.predict(rgb)
        if self.instance:
            labels = infer_instance_labels(self.instance, rgb, base.probability,
                                           device=self.base.device, side=self.side,
                                           threshold=self.threshold)
        else:
            labels = component_labels(base.alpha)
        present = sorted(int(i) for i in np.unique(labels) if i > 0)
        selected = present if keep_ids is None else sorted(set(keep_ids))
        unknown = set(selected).difference(present)
        if unknown:
            raise ValueError(f'Object IDs not found: {sorted(unknown)}; available={present}')
        # Model is not asked to guess user intention. All foreground retained by default.
        selected_mask = np.isin(labels, selected).astype(np.uint8) * 255
        if self.color:
            cleaned = apply_color_net(rgb, selected_mask, self.color, self.base.device)
        else:
            cleaned = rgb.copy()
        return FullResult(base=base, rgb=cleaned, alpha=selected_mask,
                          instance_ids=labels, selected_ids=selected,
                          instance_trained=self.instance is not None, color_trained=self.color is not None)

    def run(self, path: str | Path, output: str | Path, *, keep_ids: list[int] | None = None,
            individual: bool = False, diagnostics: bool = False) -> FullResult:
        res = self.predict(read_rgb(path), keep_ids=keep_ids)
        res.save(output, diagnostics=diagnostics, individual=individual)
        return res
