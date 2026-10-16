"""Sanity-check a YOLO dataset: counts, class balance, bad labels.

  python scripts/check_dataset.py --data data/data.yaml
"""
import argparse
from pathlib import Path

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check(data_yaml: str) -> dict:
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    names = cfg["names"]
    names = names if isinstance(names, dict) else dict(enumerate(names))
    root = Path(cfg.get("path", Path(data_yaml).parent))
    if not root.is_absolute():
        root = (Path(data_yaml).parent / root).resolve() if not root.exists() else root.resolve()
    report = {}
    for split in ("train", "val", "test"):
        if split not in cfg:
            continue
        img_dir = root / cfg[split]
        lbl_dir = Path(str(img_dir).replace("images", "labels"))
        imgs = [p for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXT] if img_dir.exists() else []
        counts = {n: 0 for n in names.values()}
        missing = empty = bad = 0
        for img in imgs:
            lbl = lbl_dir / (img.stem + ".txt")
            if not lbl.exists():
                missing += 1
                continue
            lines = [l.split() for l in lbl.read_text().splitlines() if l.strip()]
            if not lines:
                empty += 1
            for parts in lines:
                try:
                    c = int(float(parts[0]))
                    vals = [float(v) for v in parts[1:]]
                    if c not in names or any(v < 0 or v > 1 for v in vals) or len(vals) < 4:
                        bad += 1
                        continue
                    counts[names[c]] += 1
                except ValueError:
                    bad += 1
        report[split] = {"images": len(imgs), "missing_labels": missing, "empty_labels(background)": empty,
                         "bad_lines": bad, "instances": counts}
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.yaml")
    rep = check(ap.parse_args().data)
    for split, r in rep.items():
        print(f"\n[{split}] images={r['images']} missing_labels={r['missing_labels']} "
              f"background={r['empty_labels(background)']} bad_lines={r['bad_lines']}")
        for k, v in r["instances"].items():
            print(f"   {k:12s} {v}")
    low = [(s, k) for s, r in rep.items() for k, v in r["instances"].items() if s == "train" and v < 300]
    if low:
        print("\nWarning: classes with <300 training instances:", sorted({k for _, k in low}), "- consider adding data.")
