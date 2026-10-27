"""Calibrate binary alpha threshold against a held-out validation set.

Selects threshold maximizing aggregate Dice; no test images are used.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def best_from_hist(foreground_hist: np.ndarray, background_hist: np.ndarray) -> dict:
    fg = foreground_hist.astype(np.float64)
    bg = background_hist.astype(np.float64)
    tp = np.cumsum(fg[::-1])[::-1]
    fp = np.cumsum(bg[::-1])[::-1]
    fn = fg.sum() - tp
    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    best = int(np.argmax(dice[1:255]) + 1)  # reject degenerate thresholds 0 and 1
    return {'threshold': (best + .5)/256., 'dice': float(dice[best]), 'bin': best}


def calibrate(data: str, checkpoint: str, output: str, *, refiner: str | None = None,
              device: str = 'auto') -> dict:
    from cutoutnet.dataset import PairedCutoutDataset
    from cutoutnet.inference import CutoutPipeline, read_rgb
    ds = PairedCutoutDataset(data, image_size=64, augment=False)
    network = CutoutPipeline(checkpoint, refiner_checkpoint=refiner, device=device)
    pos = np.zeros(256, np.int64)
    neg = np.zeros(256, np.int64)
    for image_path, mask_path in ds.pairs:
        rgb = read_rgb(image_path)
        with Image.open(mask_path) as im:
            gt = np.asarray(ImageOps.exif_transpose(im).convert('L'), dtype=np.uint8)
        if gt.shape != rgb.shape[:2]:
            raise ValueError('GT mask size mismatch: ' + str(mask_path))
        prob = network.predict(rgb).probability
        indices = np.minimum((prob.clip(0, 1) * 255).astype(np.uint8), 255)
        pos += np.bincount(indices[gt > 127].ravel(), minlength=256)
        neg += np.bincount(indices[gt <= 127].ravel(), minlength=256)
    result = best_from_hist(pos, neg)
    result['images'] = len(ds.pairs)
    result['checkpoint'] = checkpoint
    result['refiner'] = refiner
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--refiner')
    p.add_argument('--device',default='auto')
    p.add_argument('--output',default='checkpoints/sticker/calibration.json')
    a=p.parse_args()
    calibrate(a.data,a.checkpoint,a.output,refiner=a.refiner,device=a.device)
