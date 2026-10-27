"""Save reviewed user corrections as separate future-training pairs (never auto-train)."""
from pathlib import Path
from argparse import ArgumentParser
from PIL import Image, ImageOps
import hashlib
import json
import time
import numpy as np


def register(image_path: str, corrected_mask: str, root: str = "data/cutoutnet/corrections") -> dict:
    source, mask_path = Path(image_path), Path(corrected_mask)
    with Image.open(source) as im:
        image = ImageOps.exif_transpose(im).convert("RGB")
    with Image.open(mask_path) as im:
        mask = ImageOps.exif_transpose(im).convert("L")
    if image.size != mask.size:
        raise ValueError("Correction image and mask must have identical dimensions")
    arr = np.asarray(mask)
    if not np.isin(np.unique(arr), (0, 255)).all():
        raise ValueError("Corrected mask must contain strictly 0 and 255")
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    identifier = hashlib.sha256(source.read_bytes() + mask_path.read_bytes()).hexdigest()[:20]
    img_out, mask_out = root / "images" / (identifier + ".png"), root / "masks" / (identifier + ".png")
    image.save(img_out)
    mask.save(mask_out)
    def portable(path: Path) -> str:
        try:
            return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return str(path.resolve())
    record = {"image": portable(img_out), "mask": portable(mask_out),
              "timestamp": int(time.time()), "reviewed": True}
    # Safe even when same correction is registered twice; dupe can be curated later.
    with (root / "corrections.jsonl").open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--output", default="data/cutoutnet/corrections")
    a = ap.parse_args()
    print(json.dumps(register(a.image, a.mask, a.output), ensure_ascii=False, indent=2))
