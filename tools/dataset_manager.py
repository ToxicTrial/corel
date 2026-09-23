from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CUTROOT = ROOT / "data" / "cutoutnet"
REGISTRY = CUTROOT / "registry.jsonl"
MANIFESTS = CUTROOT / "manifests"
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

SOURCES = {
    "p3m": {
        "name": "P3M-10K",
        "license": "MIT (dataset release agreement)",
        "url": "https://drive.google.com/uc?export=download&id=1LqUU7BZeiq8I3i5KxApdOJ2haXm-cEv1",
        "license_url": "https://jizhizili.github.io/files/p3m_dataset_agreement/P3M-10k_Dataset_Release_Agreement.pdf",
    },
    "aim500": {
        "name": "AIM-500",
        "license": "MIT (dataset release agreement)",
        "url": "https://drive.google.com/drive/folders/1IyPiYJUp-KtOoa-Hsm922VU3aCcidjjz?usp=sharing",
        "license_url": "https://github.com/JizhiziLi/AIM",
    },
    "am2k": {
        "name": "AM-2K",
        "license": "MIT in dataset agreement; review upstream terms before commercial use",
        "url": "https://drive.google.com/drive/folders/1SReB9Zma0TDfDhow7P5kiZNMwY9j9xMA",
        "license_url": "https://jizhizili.github.io/files/gfm_datasets_agreements/AM-2k_Dataset_Release_Agreement.pdf",
    },
}


def rel_or_abs(p: Path) -> str:
    """Store portable project-relative paths whenever possible.

    Deliberately do NOT call ``resolve()`` here: on Colab a dataset is commonly
    attached through ``data/raw/...`` symlinks. Resolving the symlink would bake
    an ephemeral ``/content/...`` target into manifests and make them invalid on
    the next cloud session.
    """
    p = p.absolute()
    root = ROOT.absolute()
    try:
        return p.relative_to(root).as_posix()
    except Exception:
        return str(p)


def resolve_path(s: str) -> Path:
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def file_stem_map(folder: Path) -> dict[str, Path]:
    out = {}
    if not folder.exists():
        return out
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out[p.stem] = p
    return out


def pair_dirs(image_dir: Path, mask_dir: Path):
    imgs = file_stem_map(image_dir)
    masks = file_stem_map(mask_dir)
    for stem in sorted(imgs.keys() & masks.keys()):
        yield stem, imgs[stem], masks[stem]


def stable_bucket(source: str, stem: str) -> float:
    h = hashlib.sha1(f"{source}:{stem}".encode()).digest()
    return int.from_bytes(h[:8], "big") / float(2**64 - 1)


def read_registry() -> list[dict]:
    if not REGISTRY.exists():
        return []
    items = []
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
    return items


def write_registry(items: list[dict]):
    CUTROOT.mkdir(parents=True, exist_ok=True)
    unique = {}
    for item in items:
        unique[item["id"]] = item
    ordered = sorted(unique.values(), key=lambda x: x["id"])
    REGISTRY.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in ordered) + ("\n" if ordered else ""), encoding="utf-8")
    print(f"Registry: {len(ordered)} pairs -> {REGISTRY}")


def add_pairs(items: list[dict], source: str, pairs, split_fn, tags: list[str], license_name: str):
    n = 0
    for stem, image, alpha in pairs:
        split = split_fn(stem)
        item = {
            "id": f"{source}:{split}:{stem}",
            "source": source,
            "split": split,
            "image": rel_or_abs(image),
            "alpha": rel_or_abs(alpha),
            "mask": rel_or_abs(alpha),
            "license": license_name,
            "tags": tags,
        }
        items.append(item)
        n += 1
    print(f"{source}: indexed {n} pairs")
    return n


def import_p3m(root: Path, items: list[dict]):
    src = SOURCES["p3m"]
    train_img = root / "train" / "blurred_image"
    train_mask = root / "train" / "mask"
    if not train_img.exists():
        raise FileNotFoundError(f"P3M train folder not found: {train_img}")

    def train_split(stem):
        return "val" if stable_bucket("p3m-train", stem) < 0.055 else "train"
    add_pairs(items, "p3m", pair_dirs(train_img, train_mask), train_split, ["portrait", "hair", "human"], src["license"])

    for sub, imgname in [("P3M-500-P", "blurred_image"), ("P3M-500-NP", "original_image")]:
        i = root / "validation" / sub / imgname
        m = root / "validation" / sub / "mask"
        if i.exists() and m.exists():
            add_pairs(items, f"p3m-{sub.lower()}", pair_dirs(i, m), lambda _s: "test", ["portrait", "human", "external-test"], src["license"])


def import_aim(root: Path, items: list[dict]):
    src = SOURCES["aim500"]
    image_dir = root / "original"
    mask_dir = root / "mask"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError("AIM-500 must contain original/ and mask/")
    add_pairs(items, "aim500", pair_dirs(image_dir, mask_dir), lambda _s: "test", ["natural", "animal", "plant", "transparent", "external-test"], src["license"])


def import_am2k(root: Path, items: list[dict]):
    src = SOURCES["am2k"]
    train_i, train_m = root / "train" / "original", root / "train" / "mask"
    val_i, val_m = root / "validation" / "original", root / "validation" / "mask"
    if train_i.exists() and train_m.exists():
        def train_split(stem):
            return "val" if stable_bucket("am2k-train", stem) < 0.10 else "train"
        add_pairs(items, "am2k", pair_dirs(train_i, train_m), train_split, ["animal", "fur", "hair"], src["license"])
    if val_i.exists() and val_m.exists():
        add_pairs(items, "am2k-validation", pair_dirs(val_i, val_m), lambda _s: "test", ["animal", "fur", "external-test"], src["license"])
    if not train_i.exists() and not val_i.exists():
        raise FileNotFoundError("AM-2K expected train/original + train/mask and/or validation/original + validation/mask")


def import_generic(images: Path, masks: Path, source: str, split: str, items: list[dict], tags: list[str]):
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    add_pairs(items, source, pair_dirs(images, masks), lambda _s: split, tags, "user-provided / verify source license")


def build_manifests(items: list[dict]):
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    by_split = {"train": [], "val": [], "test": []}
    missing = 0
    for item in items:
        if item["split"] not in by_split:
            continue
        image, mask = resolve_path(item["image"]), resolve_path(item["mask"])
        if not image.exists() or not mask.exists():
            missing += 1
            continue
        by_split[item["split"]].append(item)
    for split, rows in by_split.items():
        p = MANIFESTS / f"{split}.jsonl"
        p.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + ("\n" if rows else ""), encoding="utf-8")
        print(f"{split}: {len(rows)} -> {p}")
    if missing:
        print(f"WARNING: {missing} registry entries point to missing files")


def alpha_to_rgba(image_path: Path, alpha_path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(image_path).convert("RGB"), np.uint8)
    alpha = np.asarray(Image.open(alpha_path).convert("L"), np.uint8)
    if alpha.shape != rgb.shape[:2]:
        alpha = cv2.resize(alpha, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.dstack([rgb, alpha])


def export_cutouts(items: list[dict], max_items: int = 6000):
    out = CUTROOT / "cutouts"
    out.mkdir(parents=True, exist_ok=True)
    rows = [x for x in items if x["split"] == "train"]
    random.Random(1337).shuffle(rows)
    rows = rows[:max_items]
    ok = 0
    for idx, item in enumerate(rows):
        try:
            rgba = alpha_to_rgba(resolve_path(item["image"]), resolve_path(item.get("alpha", item["mask"])))
            if np.max(rgba[..., 3]) < 16:
                continue
            Image.fromarray(rgba).save(out / f"cut_{idx:06d}.png")
            ok += 1
        except Exception as e:
            print(f"skip {item['id']}: {e}")
    print(f"Exported {ok} RGBA cutouts -> {out}")


def add_synthetic_to_registry(items: list[dict], root: Path):
    n = add_pairs(items, "synthetic", pair_dirs(root / "images", root / "masks"), lambda stem: "val" if stable_bucket("synthetic", stem) < 0.03 else "train", ["synthetic", "multi-object", "chroma", "white-bg", "black-bg", "gradient"], "generated locally")
    return n


def _fmt_eta(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def qa(items: list[dict], sample_limit: int = 0, stats_max_side: int = 1024, progress_every: int = 10):
    rows = items[:]
    if sample_limit and len(rows) > sample_limit:
        rows = random.Random(1337).sample(rows, sample_limit)
    total = len(rows)
    if total == 0:
        print("QA: registry is empty. Run option 2 (Auto-import) first.")
        return 1

    bad = []
    fg_fracs = []
    components = []
    slow = []
    started = time.perf_counter()
    print(f"QA starting: {total} pairs | stats_max_side={stats_max_side}", flush=True)

    try:
        for idx, item in enumerate(rows, start=1):
            item_started = time.perf_counter()
            try:
                ip, mp = resolve_path(item["image"]), resolve_path(item["mask"])
                if not ip.exists():
                    bad.append((item["id"], f"image missing: {ip}")); continue
                if not mp.exists():
                    bad.append((item["id"], f"mask missing: {mp}")); continue
                with Image.open(ip) as im:
                    image_size = im.size
                    im.verify()
                with Image.open(mp) as am:
                    mask_size = am.size
                    am.verify()
                if mask_size != image_size:
                    bad.append((item["id"], f"size mismatch image={image_size} mask={mask_size}")); continue
                with Image.open(mp) as am:
                    am = am.convert("L")
                    w, h = am.size
                    scale = min(1.0, float(stats_max_side) / max(w, h)) if stats_max_side else 1.0
                    if scale < 1.0:
                        am = am.resize((max(1, int(round(w * scale))), max(1, int(round(h * scale)))), Image.Resampling.NEAREST)
                    alpha = np.asarray(am, dtype=np.uint8)
                m = alpha >= 128
                frac = float(m.mean())
                fg_fracs.append(frac)
                n, _, _, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
                components.append(max(0, n - 1))
                if frac < 0.0001 or frac > 0.9999:
                    bad.append((item["id"], f"foreground fraction {frac:.5f}"))
            except Exception as e:
                bad.append((item.get("id", "?"), str(e)))
            finally:
                dt = time.perf_counter() - item_started
                if dt >= 2.0:
                    slow.append((item.get("id", "?"), round(dt, 3)))
                if idx == 1 or idx % max(1, progress_every) == 0 or idx == total:
                    elapsed = time.perf_counter() - started
                    rate = idx / elapsed if elapsed > 0 else 0.0
                    eta = (total - idx) / rate if rate > 0 else float('inf')
                    pct = idx * 100.0 / total
                    current = item.get("id", "?")
                    print(f"QA [{idx:>5}/{total}] {pct:6.2f}% | {rate:5.1f} files/s | ETA {_fmt_eta(eta)} | bad={len(bad)} | {current}", flush=True)
    except KeyboardInterrupt:
        print("\nQA interrupted by user. Partial report will be saved.", flush=True)

    elapsed = time.perf_counter() - started
    report = {
        "checked": min(total, len(fg_fracs) + len(bad)),
        "requested": total,
        "bad_count": len(bad),
        "elapsed_seconds": round(elapsed, 3),
        "foreground_fraction": None,
        "components": None,
        "slow_files": [{"id": x, "seconds": s} for x, s in slow[:100]],
        "bad": [{"id": x, "reason": reason} for x, reason in bad[:500]],
    }
    print(f"QA checked: {len(fg_fracs) + len(bad)} / {total}")
    print(f"Bad: {len(bad)}")
    if fg_fracs:
        stats = {"median": float(np.median(fg_fracs)), "p05": float(np.quantile(fg_fracs, .05)), "p95": float(np.quantile(fg_fracs, .95))}
        report["foreground_fraction"] = stats
        print(f"FG fraction median={stats['median']:.3f}, p05={stats['p05']:.3f}, p95={stats['p95']:.3f}")
    if components:
        cstats = {"median": float(np.median(components)), "max": int(max(components))}
        report["components"] = cstats
        print(f"Components median={cstats['median']:.1f}, max={cstats['max']}")
    if slow:
        print(f"Slow files (>=2s): {len(slow)}; slowest examples:")
        for row in sorted(slow, key=lambda x: x[1], reverse=True)[:10]:
            print("SLOW", row)
    for row in bad[:20]:
        print("BAD", row)

    CUTROOT.mkdir(parents=True, exist_ok=True)
    report_path = CUTROOT / "qa_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"QA report: {report_path}")
    print(f"Elapsed: {_fmt_eta(elapsed)}")
    return len(bad)


def status(items: list[dict]):
    from collections import Counter
    print(f"Registry: {len(items)} items")
    print("By split:", dict(Counter(x["split"] for x in items)))
    print("By source:")
    for k, v in Counter(x["source"] for x in items).most_common():
        print(f"  {k}: {v}")
    print("Raw folders:")
    for name in ["P3M-10k", "AIM-500", "AM-2K"]:
        p = RAW / name
        print(f"  {name}: {'present' if any(p.rglob('*')) else 'empty'} ({p})")


def auto_import(items: list[dict]):
    errors = []
    for fn, path in [(import_p3m, RAW / "P3M-10k"), (import_aim, RAW / "AIM-500"), (import_am2k, RAW / "AM-2K")]:
        try:
            fn(path, items)
        except Exception as e:
            errors.append(str(e))
    write_registry(items)
    build_manifests(read_registry())
    if errors:
        print("\nNot imported yet:")
        for e in errors:
            print(" -", e)


def open_links():
    for src in SOURCES.values():
        print(f"Opening {src['name']} dataset and license...")
        webbrowser.open(src["license_url"])
        webbrowser.open(src["url"])


def wizard():
    print("\nCutoutNet Dataset Wizard")
    print("=========================")
    print("Recommended first dataset: P3M-10K. Then AM-2K. AIM-500 is kept for TEST only.\n")
    while True:
        print("1 - Open official download/license pages")
        print("2 - Auto-import datasets already unpacked in data/raw")
        print("3 - Show dataset status")
        print("4 - Run dataset QA")
        print("5 - Export training cutouts for synthetic generation")
        print("6 - Generate 20,000 synthetic scenes and index them")
        print("0 - Exit")
        choice = input("> ").strip()
        items = read_registry()
        if choice == "1":
            open_links()
            print("\nUnpack folders as:\n  data/raw/P3M-10k\n  data/raw/AM-2K\n  data/raw/AIM-500\n")
        elif choice == "2":
            auto_import(items)
        elif choice == "3":
            status(items)
        elif choice == "4":
            qa(items)
        elif choice == "5":
            export_cutouts(items)
        elif choice == "6":
            cutouts = CUTROOT / "cutouts"
            if not any(cutouts.glob("*.png")):
                print("No cutouts. Run option 5 first.")
                continue
            out = CUTROOT / "synthetic"
            subprocess.run([sys.executable, str(ROOT / "tools" / "build_synthetic_dataset.py"), "--count", "20000", "--out", str(out)], cwd=ROOT, check=True)
            items = read_registry()
            add_synthetic_to_registry(items, out)
            write_registry(items)
            build_manifests(read_registry())
        elif choice == "0":
            return
        else:
            print("Unknown option")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("wizard")
    sub.add_parser("status")
    sub.add_parser("open-links")
    sub.add_parser("auto-import")
    q = sub.add_parser("qa"); q.add_argument("--sample-limit", type=int, default=0); q.add_argument("--stats-max-side", type=int, default=1024); q.add_argument("--progress-every", type=int, default=10)
    e = sub.add_parser("export-cutouts"); e.add_argument("--max-items", type=int, default=6000)
    g = sub.add_parser("import-generic")
    g.add_argument("--images", required=True); g.add_argument("--masks", required=True); g.add_argument("--source", required=True); g.add_argument("--split", default="train"); g.add_argument("--tags", default="")
    args = ap.parse_args()
    items = read_registry()
    if args.cmd in {None, "wizard"}: wizard()
    elif args.cmd == "status": status(items)
    elif args.cmd == "open-links": open_links()
    elif args.cmd == "auto-import": auto_import(items)
    elif args.cmd == "qa": raise SystemExit(qa(items, args.sample_limit, args.stats_max_side, args.progress_every))
    elif args.cmd == "export-cutouts": export_cutouts(items, args.max_items)
    elif args.cmd == "import-generic":
        import_generic(Path(args.images), Path(args.masks), args.source, args.split, items, [x.strip() for x in args.tags.split(',') if x.strip()])
        write_registry(items); build_manifests(read_registry())

if __name__ == "__main__":
    main()
