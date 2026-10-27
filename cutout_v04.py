"""End-to-end CLI for pretrained CutoutNet + optional trained auxiliary CNNs."""
from pathlib import Path
import argparse
from cutoutnet.full_pipeline import FullPipeline
from cutoutnet.sticker import save_cut_svg

EXTS = {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}


def main():
    ap = argparse.ArgumentParser(description='CutoutNet v0.4 full sticker pipeline')
    ap.add_argument('input', help='Input image or folder')
    ap.add_argument('--checkpoint', required=True, help='Base CutoutNet pretrained weights')
    ap.add_argument('--refiner', help='TRAINED high-resolution ROI Refiner checkpoint')
    ap.add_argument('--instances', help='TRAINED instance segmentation CNN checkpoint')
    ap.add_argument('--color', help='TRAINED RGB edge-decontamination CNN checkpoint')
    ap.add_argument('--device', choices=('auto','cpu','cuda'), default='auto')
    ap.add_argument('--image-size', type=int)
    ap.add_argument('--threshold', type=float, default=.5)
    ap.add_argument('--calibration', help='JSON created by calibrate_v04.py')
    ap.add_argument('--max-rois', type=int, default=80)
    ap.add_argument('--instance-side', type=int, default=640)
    ap.add_argument('--output', default='outputs/v04')
    ap.add_argument('--export-instances', action='store_true', help='Export a PNG for each object')
    ap.add_argument('--keep-ids', help='Comma-separated IDs of objects to keep; otherwise keep all')
    ap.add_argument('--diagnostics', action='store_true')
    ap.add_argument('--cut-svg', action='store_true')
    ap.add_argument('--dpi', type=float, default=300)
    ap.add_argument('--offset-mm', type=float, default=2.0)
    ap.add_argument('--cut-mode', choices=('separate','unified'), default='separate')
    args = ap.parse_args()
    selected = [int(x) for x in args.keep_ids.split(',') if x.strip()] if args.keep_ids else None
    if args.calibration:
        import json
        args.threshold = float(json.loads(Path(args.calibration).read_text(encoding='utf-8'))['threshold'])
    pipe = FullPipeline(args.checkpoint, refiner_checkpoint=args.refiner,
                        instance_checkpoint=args.instances, color_checkpoint=args.color,
                        device=args.device, image_size=args.image_size, threshold=args.threshold,
                        max_rois=args.max_rois, instance_side=args.instance_side)
    inp = Path(args.input)
    if not inp.exists():
        ap.error(f'File or folder not found: {inp}')
    images = [inp] if inp.is_file() else sorted(x for x in inp.iterdir() if x.suffix.lower() in EXTS)
    if not images:
        ap.error('No images found')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(images):
        name = f'{image.stem}_cutout.png'
        if i and any(p.stem == image.stem for p in images[:i]):
            name = f'{image.stem}_{i:04}_cutout.png'
        dest = out / name
        res = pipe.run(image, dest, keep_ids=selected,
                       individual=args.export_instances, diagnostics=args.diagnostics)
        print(f'{image} -> {dest}; objects={len(res.selected_ids)}; '
              f'ROI={len(res.base.regions)}; trained instance={res.instance_trained}; '
              f'trained color={res.color_trained}')
        if args.cut_svg and res.alpha.any():
            svg = dest.with_suffix('.svg')
            n = save_cut_svg(res.alpha, svg, dpi=args.dpi, offset_mm=args.offset_mm,
                             mode=args.cut_mode)
            print(f'Cut lines: {svg} ({n})')


if __name__ == '__main__':
    main()
