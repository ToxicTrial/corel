"""CLI for actual v0.2/v0.3 CutoutNet trained checkpoints and sticker SVG."""
from pathlib import Path
import argparse

from cutoutnet.inference import CutoutPipeline
from cutoutnet.sticker import save_cut_svg

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> None:
    p = argparse.ArgumentParser(description="CutoutNet v0.3 local image cutout & cut-line")
    p.add_argument("input", help="Input image file or directory")
    p.add_argument("--checkpoint", required=True, help="Existing v0.2 best.pt or last_full.pt")
    p.add_argument("--output", default="outputs", help="Output directory")
    p.add_argument("--refiner", default=None, help="Optional TRAINED ROI refiner best.pt")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--threshold", type=float, default=.5)
    p.add_argument("--patch", type=int, default=512)
    p.add_argument("--overlap", type=int, default=128)
    p.add_argument("--max-rois", type=int, default=80)
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--cut-svg", action="store_true", help="Export physical cut paths")
    p.add_argument("--offset-mm", type=float, default=2.0)
    p.add_argument("--dpi", type=float, default=300.0)
    p.add_argument("--cut-mode", choices=("separate", "unified"), default="separate")
    args = p.parse_args()
    pipeline = CutoutPipeline(args.checkpoint, refiner_checkpoint=args.refiner,
                             device=args.device, image_size=args.image_size,
                             threshold=args.threshold, patch=args.patch,
                             overlap=args.overlap, max_rois=args.max_rois)
    inp, out = Path(args.input), Path(args.output)
    if not inp.exists():
        p.error(f"Input does not exist: {inp}")
    paths = [inp] if inp.is_file() else sorted(x for x in inp.iterdir() if x.suffix.lower() in EXTS)
    if not paths:
        p.error("No supported images found")
    out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        destination = out / (path.stem + "_cutout.png")
        result = pipeline.run(path, destination, save_diagnostics=args.diagnostics)
        print(f"{path.name} -> {destination} | epoch={result.checkpoint_epoch} ROI={len(result.regions)}")
        if args.cut_svg:
            vector = out / (path.stem + "_cutline.svg")
            n = save_cut_svg(result.alpha, vector, offset_mm=args.offset_mm,
                             dpi=args.dpi, mode=args.cut_mode)
            print(f"Cut path: {vector} | contours={n}")


if __name__ == "__main__":
    main()
