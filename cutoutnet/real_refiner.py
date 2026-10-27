"""Fine-tune ROIRefiner on ACTUAL mistakes of a frozen CutoutNet.

Two commands: cache real full-resolution coarse masks; train on those errors.
Train and validation manifests are separate; never cache held-out test into train.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from .dataset import PairedCutoutDataset
from .model import ROIRefiner
from .losses import dice_loss
from .inference import CutoutPipeline, read_rgb


def cache_actual_errors(dataset: str, checkpoint: str, output: str,
                        device: str = 'auto') -> None:
    ds = PairedCutoutDataset(dataset, image_size=64, augment=False)
    net = CutoutPipeline(checkpoint, device=device)
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for image_path, mask_path in ds.pairs:
        src = read_rgb(image_path)
        with Image.open(mask_path) as im:
            gt = np.asarray(ImageOps.exif_transpose(im).convert('L'), dtype=np.uint8).copy()
        if gt.shape != src.shape[:2]:
            raise ValueError(f'Mask size mismatch: {mask_path}')
        result = net.predict(src)
        identifier = hashlib.sha256(str(image_path.resolve()).encode('utf-8')).hexdigest()[:24]
        cache_path = folder / (identifier + '.npz')
        tmp_path = folder / (identifier + '.tmp.npz')
        np.savez_compressed(tmp_path, probability=result.probability.astype(np.float16),
                            risk=result.risk.astype(np.float16))
        os.replace(tmp_path, cache_path)
        rows.append({'image': str(image_path.resolve()), 'mask': str(mask_path.resolve()),
                     'cache': str(cache_path.resolve())})
    manifest = folder / 'manifest.jsonl'
    manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
    print(f'{len(rows)} real-error pairs saved in {manifest}')


class RealErrorDataset(Dataset):
    def __init__(self, manifest: str, crop: int = 512, repeat: int = 6,
                 augment: bool = True):
        self.rows = [json.loads(line) for line in Path(manifest).read_text(encoding='utf-8').splitlines() if line.strip()]
        if not self.rows:
            raise RuntimeError('Empty error cache manifest')
        self.crop, self.repeat, self.augment = crop, max(1, repeat if augment else 1), augment

    def __len__(self):
        return len(self.rows) * self.repeat

    def __getitem__(self, index):
        row = self.rows[index % len(self.rows)]
        with Image.open(row['image']) as f:
            rgb = np.asarray(ImageOps.exif_transpose(f).convert('RGB'), dtype=np.uint8).copy()
        with Image.open(row['mask']) as f:
            gt = np.asarray(ImageOps.exif_transpose(f).convert('L'), dtype=np.uint8).copy()
        with np.load(row['cache']) as f:
            coarse, risk = np.asarray(f['probability'], np.float32), np.asarray(f['risk'], np.float32)
        if coarse.shape != rgb.shape[:2] or gt.shape != coarse.shape or risk.shape != coarse.shape:
            raise ValueError('Stale or mismatched error-cache entry: ' + str(row['cache']))
        h, w = gt.shape
        errors = ((coarse >= .5) != (gt >= 128)).astype(np.uint8)
        hard = cv2.dilate(errors, np.ones((7, 7), np.uint8))
        if not hard.any():
            hard = cv2.morphologyEx((gt >= 128).astype(np.uint8), cv2.MORPH_GRADIENT,
                                    np.ones((5, 5), np.uint8))
        ys, xs = np.where(hard > 0)
        rng = np.random.default_rng(None if self.augment else index)
        if len(xs) and rng.random() < .9:
            ix = int(rng.integers(0, len(xs)))
            cy, cx = int(ys[ix]), int(xs[ix])
        else:
            cy, cx = int(rng.integers(h)), int(rng.integers(w))
        s = self.crop
        y0, x0 = max(0, min(h-s, cy-s//2)), max(0, min(w-s, cx-s//2))
        rgb, gt = rgb[y0:y0+s, x0:x0+s], gt[y0:y0+s, x0:x0+s]
        coarse, risk = coarse[y0:y0+s, x0:x0+s], risk[y0:y0+s, x0:x0+s]
        dh, dw = max(0, s-rgb.shape[0]), max(0, s-rgb.shape[1])
        rgb = cv2.copyMakeBorder(rgb, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        gt = cv2.copyMakeBorder(gt, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=0)
        coarse = cv2.copyMakeBorder(coarse, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=0)
        risk = cv2.copyMakeBorder(risk, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=0)
        if self.augment and rng.random() < .5:
            rgb, gt, coarse, risk = (v[:, ::-1].copy() for v in (rgb, gt, coarse, risk))
        return {'rgb': torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float() / 255,
                'gt': torch.from_numpy((gt >= 128).astype(np.float32)[None].copy()),
                'coarse': torch.from_numpy(coarse[None].copy().clip(1e-4, 1-1e-4)),
                'uncertainty': torch.from_numpy(risk[None].copy())}


def _loss(net: ROIRefiner, data: dict, *, with_grad: bool = True):
    low = torch.logit(data['coarse'].clamp(1e-4, 1-1e-4))
    delta = net(data['rgb'], data['coarse'], data['uncertainty'])
    pred = low + delta
    # Same decision boundary and residual-logit representation as inference.
    weights = 1 + 2 * data['uncertainty']
    bce = F.binary_cross_entropy_with_logits(pred.float(), data['gt'], reduction='none')
    return (bce * weights).mean() + dice_loss(pred, data['gt'])


def train_actual_errors(train_manifest: str, val_manifest: str, output: str,
                        *, init: str | None = None, base: int = 48, crop: int = 512,
                        epochs: int = 20, batch_size: int = 2,
                        lr: float = 5e-5, device: str = 'auto'):
    dev = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device))
    tl = DataLoader(RealErrorDataset(train_manifest, crop, augment=True), batch_size, shuffle=True)
    vl = DataLoader(RealErrorDataset(val_manifest, crop, augment=False), batch_size)
    model = ROIRefiner(base=base).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    best, start = float('inf'), 1
    if init:
        ck = torch.load(init, map_location='cpu', weights_only=False)
        if ck.get('format') != 'cutoutnet-refiner-v1' or ck['base'] != base:
            raise ValueError('Initialization is not compatible ROIRefiner')
        model.load_state_dict(ck['model'])
    resume = folder / 'last_full.pt'
    if resume.exists():
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        if ck.get('base') != base or ck.get('crop') != crop:
            raise ValueError('Checkpoint dimensions mismatch')
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        best, start = ck['best_val_loss'], ck['epoch'] + 1
    for epoch in range(start, epochs + 1):
        model.train()
        ls = []
        for b in tl:
            b = {k: v.to(dev) for k, v in b.items()}
            loss = _loss(model, b)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite refiner loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            ls.append(loss.item())
        model.eval()
        validation, baseline = [], []
        with torch.inference_mode():
            for b in vl:
                b = {k: v.to(dev) for k, v in b.items()}
                validation.append(_loss(model, b).item())
                base_logits = torch.logit(b['coarse'].clamp(1e-4, 1-1e-4))
                base_bce = F.binary_cross_entropy_with_logits(base_logits, b['gt'], reduction='none')
                baseline.append(((base_bce * (1 + 2 * b['uncertainty'])).mean() + dice_loss(base_logits, b['gt'])).item())
        score = float(np.mean(validation))
        base_score = float(np.mean(baseline))
        print(json.dumps({'epoch': epoch, 'train': float(np.mean(ls)), 'validation': score,
                          'no_refiner': base_score, 'improves_loss': score < base_score}))
        common = {'format': 'cutoutnet-refiner-v1', 'model': model.state_dict(), 'base': base,
                  'crop': crop, 'epoch': epoch, 'real_errors': True}
        if score < best:
            best = score
            tmp = folder / 'best.pt.tmp'
            torch.save({**common, 'baseline_loss': base_score, 'val_loss': score}, tmp)
            os.replace(tmp, folder / 'best.pt')
        tmp = folder / 'last_full.pt.tmp'
        torch.save({**common, 'optimizer': opt.state_dict(), 'best_val_loss': best}, tmp)
        os.replace(tmp, resume)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    cache = sub.add_parser('cache')
    cache.add_argument('--dataset', required=True)
    cache.add_argument('--checkpoint', required=True)
    cache.add_argument('--output', required=True)
    cache.add_argument('--device', default='auto')
    tr = sub.add_parser('train')
    tr.add_argument('--train-manifest', required=True)
    tr.add_argument('--val-manifest', required=True)
    tr.add_argument('--output', default='checkpoints/refiner_real')
    tr.add_argument('--init')
    tr.add_argument('--base', type=int, default=48)
    tr.add_argument('--crop', type=int, default=512)
    tr.add_argument('--epochs', type=int, default=20)
    tr.add_argument('--batch-size', type=int, default=2)
    tr.add_argument('--lr', type=float, default=5e-5)
    tr.add_argument('--device', default='auto')
    a = p.parse_args()
    if a.action == 'cache':
        cache_actual_errors(a.dataset, a.checkpoint, a.output, a.device)
    else:
        train_actual_errors(a.train_manifest, a.val_manifest, a.output,
                            init=a.init, base=a.base, crop=a.crop, epochs=a.epochs,
                            batch_size=a.batch_size, lr=a.lr, device=a.device)

if __name__ == '__main__':
    main()
