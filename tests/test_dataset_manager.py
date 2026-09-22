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
