"""Split a flat folder of images + YOLO labels into train/valid/test.

  python scripts/split_dataset.py --images raw/images --labels raw/labels --out data/merged
"""
import argparse
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split(images, labels, out, ratios=(0.7, 0.2, 0.1), seed=42):
    imgs = sorted(p for p in Path(images).iterdir() if p.suffix.lower() in IMG_EXT)
    random.Random(seed).shuffle(imgs)
    n = len(imgs)
    n_train, n_val = int(n * ratios[0]), int(n * ratios[1])
    parts = {"train": imgs[:n_train], "valid": imgs[n_train:n_train + n_val], "test": imgs[n_train + n_val:]}
    for name, items in parts.items():
        (Path(out) / name / "images").mkdir(parents=True, exist_ok=True)
        (Path(out) / name / "labels").mkdir(parents=True, exist_ok=True)
        for img in items:
            shutil.copy2(img, Path(out) / name / "images" / img.name)
            lbl = Path(labels) / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, Path(out) / name / "labels" / lbl.name)
    return {k: len(v) for k, v in parts.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.2, 0.1])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(split(a.images, a.labels, a.out, a.ratios, a.seed))
