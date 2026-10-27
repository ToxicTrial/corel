"""Fine-tune the primary CutoutNet with boundary + topology losses.

Requires the patched cutoutnet/trainer.py in the v0.4 release.
"""
import argparse
from cutoutnet.trainer import train

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config/train_sticker.yaml')
    args = p.parse_args()
    train(args.config)
