import json
from pathlib import Path
import numpy as np
from PIL import Image

from cutoutnet.dataset import PairedCutoutDataset


def test_cutoutnet_dataset_reads_jsonl_manifest(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path/'raw').mkdir()
    rgb = np.zeros((20,30,3), np.uint8); rgb[3:15,4:20] = (120,50,200)
    mask = np.zeros((20,30), np.uint8); mask[3:15,4:20] = 255
    Image.fromarray(rgb).save(tmp_path/'raw'/'a.jpg')
    Image.fromarray(mask).save(tmp_path/'raw'/'a.png')
    manifest = tmp_path/'train.jsonl'
    manifest.write_text(json.dumps({'image':'raw/a.jpg','mask':'raw/a.png'})+'\n', encoding='utf-8')
    ds = PairedCutoutDataset(manifest, image_size=64, augment=False)
    assert len(ds) == 1
    item = ds[0]
    assert tuple(item['image'].shape) == (3,64,64)
    assert tuple(item['targets']['final'].shape) == (1,64,64)


def test_cutoutnet_dataset_remaps_stale_windows_manifest_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    img_dir = tmp_path / "data" / "raw" / "P3M-10k" / "train" / "blurred_image"
    mask_dir = tmp_path / "data" / "raw" / "P3M-10k" / "train" / "mask"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    rgb = np.zeros((20, 30, 3), np.uint8)
    rgb[3:15, 4:20] = (120, 50, 200)
    mask = np.zeros((20, 30), np.uint8)
    mask[3:15, 4:20] = 255
    Image.fromarray(rgb).save(img_dir / "a.jpg")
    Image.fromarray(mask).save(mask_dir / "a.png")
    manifest = tmp_path / "train.jsonl"
    row = {
        "image": r"C:\old\CutoutLab\data\raw\P3M-10k\train\blurred_image\a.jpg",
        "mask": r"C:\old\CutoutLab\data\raw\P3M-10k\train\mask\a.png",
    }
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    ds = PairedCutoutDataset(manifest, image_size=64, augment=False)
    assert len(ds) == 1
