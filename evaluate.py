import argparse
from cutoutnet.evaluate import evaluate

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/cutoutnet/best.pt")
    ap.add_argument("--data", default="data/cutoutnet/manifests/val.jsonl")
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    evaluate(args.checkpoint, args.data, args.image_size, args.batch_size, args.device)
