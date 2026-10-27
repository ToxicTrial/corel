"""Train InstanceNet on images plus true per-object integer ID masks.

Folder format: ROOT/images/name.jpg and ROOT/instances/name.png (16-bit PNG
supported). 0=background and 1,2,... are distinct physical objects.
Binary masks cannot teach the network to split touching objects.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader

from .instances import InstanceNet, make_instance_targets, instance_loss


class InstanceDataset(Dataset):
    def __init__(self, root: str | Path, size: int = 512, augment: bool = False):
        root = Path(root)
        self.items = []
        masks = {p.stem: p for p in (root / 'instances').glob('*.png')}
        for im in sorted((root / 'images').iterdir()):
            if im.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'} and im.stem in masks:
                self.items.append((im, masks[im.stem]))
        if not self.items:
            raise RuntimeError(f'No image/instance-ID pairs under {root}')
        self.size, self.augment = size, augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        image_path, mask_path = self.items[i]
        with Image.open(image_path) as f:
            rgb = np.array(ImageOps.exif_transpose(f).convert('RGB'))
        with Image.open(mask_path) as f:
            ids = np.array(f)
        if ids.ndim != 2 or ids.shape != rgb.shape[:2] or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(f'Invalid ID mask: {mask_path}')
        h, w = ids.shape
        ratio = min(self.size / h, self.size / w)
        nh, nw = max(1, round(h * ratio)), max(1, round(w * ratio))
        rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        ids = cv2.resize(ids.astype(np.int32), (nw, nh), interpolation=cv2.INTER_NEAREST)
        image = np.zeros((self.size, self.size, 3), np.uint8)
        label = np.zeros((self.size, self.size), np.int32)
        y0, x0 = (self.size - nh) // 2, (self.size - nw) // 2
        image[y0:y0+nh, x0:x0+nw] = rgb
        label[y0:y0+nh, x0:x0+nw] = ids
        if self.augment and np.random.random() < .5:
            image, label = image[:, ::-1].copy(), label[:, ::-1].copy()
        if self.augment and np.random.random() < .5:
            image, label = image[::-1].copy(), label[::-1].copy()
        center, offset, foreground = make_instance_targets(label)
        return {
            'rgb': torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255,
            'fg': torch.from_numpy(foreground),
            'center': torch.from_numpy(center),
            'offset': torch.from_numpy(offset),
        }


def train(train_root: str, val_root: str, output: str, *, size: int = 512,
          base: int = 24, batch_size: int = 2, epochs: int = 40,
          lr: float = 1e-4, device: str = 'auto') -> None:
    dev = torch.device('cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device))
    if dev.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    ds = InstanceDataset(train_root, size, augment=True)
    valid = InstanceDataset(val_root, size, augment=False)
    tl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    vl = DataLoader(valid, batch_size=batch_size)
    net = InstanceNet(base).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    resume = path / 'last_full.pt'
    start, best = 1, float('inf')
    if resume.exists():
        state = torch.load(resume, map_location='cpu', weights_only=False)
        if state['base'] != base or state['size'] != size:
            raise ValueError('Resume instance model/size mismatch')
        net.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        start, best = state['epoch'] + 1, state['best_loss']
    for epoch in range(start, epochs + 1):
        net.train()
        tr = []
        for b in tl:
            b = {k: v.to(dev) for k, v in b.items()}
            pred = net(b['rgb'], b['fg'])
            loss = instance_loss(pred, b['center'], b['offset'], b['fg'])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite instance loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
            tr.append(loss.item())
        net.eval()
        val = []
        with torch.inference_mode():
            for b in vl:
                b = {k: v.to(dev) for k, v in b.items()}
                val.append(instance_loss(net(b['rgb'], b['fg']), b['center'], b['offset'], b['fg']).item())
        score = float(np.mean(val))
        print(json.dumps({'epoch': epoch, 'train_loss': float(np.mean(tr)), 'val_loss': score}))
        common = {'format': 'cutoutnet-instance-v1', 'model': net.state_dict(),
                  'base': base, 'size': size, 'epoch': epoch}
        if score < best:
            best = score
            tmp = path / 'best.pt.tmp'
            torch.save(common, tmp)
            os.replace(tmp, path / 'best.pt')
        tmp = path / 'last_full.pt.tmp'
        torch.save({**common, 'optimizer': opt.state_dict(), 'best_loss': best}, tmp)
        os.replace(tmp, resume)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True)
    p.add_argument('--val', required=True)
    p.add_argument('--output', default='checkpoints/instance')
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--base', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default='auto')
    a = p.parse_args()
    train(a.train, a.val, a.output, size=a.size, base=a.base, batch_size=a.batch_size,
          epochs=a.epochs, lr=a.lr, device=a.device)

if __name__ == '__main__':
    main()
