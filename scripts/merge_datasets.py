"""Merge several YOLO-format datasets into one with unified class IDs.

Why: oil, algae and plastic usually come from different datasets with different
class names/IDs. This remaps them to: 0 plastic, 1 oil_spill, 2 micro_algae.

Config (JSON), e.g. data/merge_config.json:
{
  "output": "data/merged",
  "classes": ["plastic", "oil_spill", "micro_algae"],
  "sources": [
    {"path": "data/raw/plastic_ds", "map": {"trash": "plastic", "bottle": "plastic"}},
    {"path": "data/raw/oil_ds",     "map": {"oil": "oil_spill"}},
    {"path": "data/raw/algae_ds",   "map": {"algae": "micro_algae", "bloom": "micro_algae"}}
  ]
}
Each source must have data.yaml and train|valid(val)|test folders with images/ and labels/.
Source classes that are not in "map" are dropped. Images with no labels are kept
as background (negative) samples.

  python scripts/merge_datasets.py --config data/merge_config.json
"""
import argparse
import json
import shutil
from pathlib import Path

import yaml

SPLITS = {"train": ["train"], "valid": ["valid", "val"], "test": ["test"]}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_names(root: Path):
    cfg = yaml.safe_load((root / "data.yaml").read_text())
    names = cfg["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def find_split(root: Path, aliases):
    for a in aliases:
        if (root / a / "images").is_dir():
            return root / a
    return None


def merge(config: dict):
    out = Path(config["output"])
    classes = config["classes"]
    cls_idx = {c: i for i, c in enumerate(classes)}
    stats = {c: 0 for c in classes}
    n_images = 0

    for split in SPLITS:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    for si, src in enumerate(config["sources"]):
        root = Path(src["path"])
        names = load_names(root)
        id_map = {}
        for i, n in enumerate(names):
            target = src["map"].get(n)
            if target is not None:
                if target not in cls_idx:
                    raise ValueError(f"'{target}' is not in classes {classes}")
                id_map[i] = cls_idx[target]
        print(f"[{root.name}] source classes {names} -> mapped ids {id_map}")

        for split, aliases in SPLITS.items():
            sdir = find_split(root, aliases)
            if sdir is None:
                continue
            for img in (sdir / "images").iterdir():
                if img.suffix.lower() not in IMG_EXT:
                    continue
                lbl = sdir / "labels" / (img.stem + ".txt")
                lines_out, had_lines = [], False
                if lbl.exists():
                    for line in lbl.read_text().splitlines():
                        parts = line.split()
                        if not parts:
                            continue
                        had_lines = True
                        c = int(float(parts[0]))
                        if c in id_map:
                            lines_out.append(" ".join([str(id_map[c])] + parts[1:]))
                if had_lines and not lines_out:
                    continue  # only contained dropped classes -> skip image
                new_stem = f"s{si}_{img.stem}"
                shutil.copy2(img, out / split / "images" / (new_stem + img.suffix))
                (out / split / "labels" / (new_stem + ".txt")).write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
                n_images += 1
                for l in lines_out:
                    stats[classes[int(l.split()[0])]] += 1

    data_yaml = {
        "path": str(out.resolve()), "train": "train/images", "val": "valid/images", "test": "test/images",
        "nc": len(classes), "names": {i: c for i, c in enumerate(classes)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    print(f"Merged {n_images} images -> {out}")
    print("Instances per class:", stats)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    merge(json.loads(Path(ap.parse_args().config).read_text()))
