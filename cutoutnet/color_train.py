"""Train color decontamination on CLEAN RGBA cutouts with synthetic backgrounds.

train/val directories contain RGBA PNGs with foreground RGB valid inside alpha.
No synthetic method guarantees removal of all real photographic color spill.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from .colors import ColorCleanNet


class ColorDataset(Dataset):
    def __init__(self, folder: str, crop: int = 256, augment: bool = False):
        self.images = sorted(Path(folder).glob('*.png'))
        if not self.images:
            raise RuntimeError(f'No RGBA PNG cutouts in {folder}')
        self.crop, self.augment = crop, augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        rng = np.random.default_rng(idx if not self.augment else None)
        with Image.open(self.images[idx]) as im:
            if 'A' not in im.convert('RGBA').getbands() or im.mode not in ('RGBA', 'LA', 'P'):
                raise ValueError(f'Expected RGBA cutout: {self.images[idx]}')
            rgba = np.asarray(im.convert('RGBA')).copy()
        h, w = rgba.shape[:2]
        fg = (rgba[:, :, 3] > 127).astype(np.uint8)
        edge = cv2.morphologyEx(fg, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
        ys, xs = np.where(edge > 0)
        if len(xs):
            k = int(rng.integers(len(xs)))
            cy, cx = int(ys[k]), int(xs[k])
        else:
            cy, cx = h // 2, w // 2
        s = self.crop
        x0, y0 = max(0, min(w - s, cx - s // 2)), max(0, min(h - s, cy - s // 2))
        rgba = rgba[y0:y0+s, x0:x0+s]
        rgba = cv2.copyMakeBorder(rgba, 0, max(0, s-rgba.shape[0]), 0, max(0, s-rgba.shape[1]), cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        clean = rgba[:, :, :3].astype(np.float32) / 255
        alpha = (rgba[:, :, 3] > 127).astype(np.float32)
        smooth = cv2.GaussianBlur(alpha, (5, 5), sigmaX=float(rng.uniform(.7, 2.0)))
        bg_color = rng.uniform(0, 1, (1, 1, 3)).astype(np.float32)
        bg_color2 = rng.uniform(0, 1, (1, 1, 3)).astype(np.float32)
        bg = np.linspace(0, 1, s, dtype=np.float32)[None, :, None] * bg_color + (1-np.linspace(0, 1, s, dtype=np.float32))[None, :, None] * bg_color2
        bg = np.broadcast_to(bg, (s, s, 3))
        dirty = np.clip(clean * smooth[:, :, None] + bg * (1 - smooth[:, :, None]), 0, 1)
        if self.augment:
            dirty = np.clip(dirty + rng.normal(0, .012, dirty.shape), 0, 1)
        return {'dirty': torch.from_numpy(dirty.transpose(2, 0, 1).copy()).float(),
                'clean': torch.from_numpy(clean.transpose(2, 0, 1).copy()).float(),
                'alpha': torch.from_numpy(alpha[None].copy()).float()}


def train(train_dir: str, val_dir: str, out_dir: str, *, crop: int = 256, base: int = 24,
          batch_size: int = 4, epochs: int = 30, lr: float = 1e-4, device: str = 'auto'):
    dev = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device))
    tl = DataLoader(ColorDataset(train_dir, crop, True), batch_size, shuffle=True)
    vl = DataLoader(ColorDataset(val_dir, crop, False), batch_size)
    net = ColorCleanNet(base).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    start, best = 1, float('inf')
    resume = path / 'last_full.pt'
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['base'] != base or state['crop'] != crop:
            raise ValueError('Color checkpoint configuration mismatch')
        net.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        start, best = state['epoch'] + 1, state['best_loss']
    for epoch in range(start, epochs + 1):
        net.train()
        trainloss = []
        for batch in tl:
            b = {k: v.to(dev) for k, v in batch.items()}
            pred = net(b['dirty'], b['alpha'])
            a = b['alpha']
            blurred = F.avg_pool2d(a, 11, stride=1, padding=5)
            band = a * (blurred < .999).float()
            weights = a + 4 * band
            loss = (torch.abs(pred - b['clean']) * weights).sum() / (3 * weights.sum() + 1e-6)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite color training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            trainloss.append(float(loss.item()))
        net.eval()
        losses = []
        with torch.inference_mode():
            for batch in vl:
                b = {k: v.to(dev) for k, v in batch.items()}
                p = net(b['dirty'], b['alpha'])
                l = (torch.abs(p - b['clean']) * b['alpha']).sum() / (3 * b['alpha'].sum() + 1e-6)
                losses.append(l.item())
        score = float(np.mean(losses))
        print(json.dumps({'epoch': epoch, 'train_loss': float(np.mean(trainloss)), 'val_loss': score}))
        common = {'format': 'cutoutnet-color-v1', 'model': net.state_dict(), 'base': base, 'crop': crop, 'epoch': epoch}
        if score < best:
            best = score
            temp = path / 'best.pt.tmp'
            torch.save(common, temp)
            os.replace(temp, path / 'best.pt')
        temp = path / 'last_full.pt.tmp'
        torch.save({**common, 'optimizer': opt.state_dict(), 'best_loss': best}, temp)
        os.replace(temp, resume)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True)
    p.add_argument('--val', required=True)
    p.add_argument('--output', default='checkpoints/color')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--crop', type=int, default=256)
    p.add_argument('--base', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default='auto')
    a = p.parse_args()
    train(a.train, a.val, a.output, crop=a.crop, base=a.base,
          batch_size=a.batch_size, epochs=a.epochs, lr=a.lr, device=a.device)

if __name__ == '__main__':
    main()
