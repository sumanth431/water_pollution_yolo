"""Tests for the dataset tools and helper functions (no GPU / torch needed).

Run: python -m pytest tests -q   (or: python tests/test_pipeline.py)
"""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_dataset import check  # noqa: E402
from scripts.merge_datasets import merge  # noqa: E402
from scripts.split_dataset import split  # noqa: E402
from src.utils import build_alerts, counts_from_result, coverage_from_result  # noqa: E402


def _make_ds(root: Path, names, items):
    """items: {split: [(stem, [label lines])]}"""
    (root).mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text(yaml.safe_dump({"names": names}))
    for split, lst in items.items():
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir(parents=True)
        for stem, lines in lst:
            (root / split / "images" / f"{stem}.jpg").write_bytes(b"x")
            (root / split / "labels" / f"{stem}.txt").write_text("\n".join(lines))


def test_merge_and_check():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        _make_ds(t / "a", ["trash", "fish"], {"train": [("i1", ["0 .5 .5 .2 .2", "1 .3 .3 .1 .1"]), ("i2", ["1 .5 .5 .2 .2"]), ("i3", [])],
                                              "valid": [("v1", ["0 .5 .5 .2 .2"])]})
        _make_ds(t / "b", ["oil"], {"train": [("o1", ["0 .5 .5 .4 .4"])], "val": [("ov", ["0 .5 .5 .4 .4"])], "test": [("ot", ["0 .5 .5 .4 .4"])]})
        cfg = {"output": str(t / "out"), "classes": ["plastic", "oil_spill", "micro_algae"],
               "sources": [{"path": str(t / "a"), "map": {"trash": "plastic"}}, {"path": str(t / "b"), "map": {"oil": "oil_spill"}}]}
        stats = merge(cfg)
        assert stats == {"plastic": 2, "oil_spill": 3, "micro_algae": 0}
        # i2 only had a dropped class -> skipped; i3 background kept
        assert not (t / "out/train/images/s0_i2.jpg").exists()
        assert (t / "out/train/images/s0_i3.jpg").exists()
        rep = check(str(t / "out/data.yaml"))
        assert rep["train"]["bad_lines"] == 0 and rep["train"]["images"] == 3


def test_split():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "im").mkdir(); (t / "lb").mkdir()
        for i in range(10):
            (t / "im" / f"{i}.jpg").write_bytes(b"x")
            (t / "lb" / f"{i}.txt").write_text("0 .5 .5 .1 .1")
        assert split(t / "im", t / "lb", t / "o") == {"train": 7, "valid": 2, "test": 1}


def _fake_result():
    return SimpleNamespace(
        boxes=SimpleNamespace(cls=np.array([0, 0, 1]), xyxy=np.array([[0, 0, 100, 100], [0, 0, 50, 50], [0, 0, 320, 240]]), __len__=lambda s: 3),
        orig_shape=(480, 640), masks=None)


class _R(SimpleNamespace):
    pass


def test_utils():
    r = _fake_result()
    r.boxes = type("B", (), {"cls": np.array([0, 0, 1]), "xyxy": np.array([[0, 0, 100, 100], [0, 0, 50, 50], [0, 0, 320, 240]]), "__len__": lambda s: 3})()
    names = {0: "plastic", 1: "oil_spill", 2: "micro_algae"}
    counts = counts_from_result(r, names)
    assert counts == {"plastic": 2, "oil_spill": 1}
    cov = coverage_from_result(r, names)
    assert abs(cov["oil_spill"] - 25.0) < 1e-6
    alerts = build_alerts(counts)
    assert alerts[0][0] == "error" and "Oil spill" in alerts[0][1]
    assert build_alerts({})[0][0] == "success"


if __name__ == "__main__":
    test_merge_and_check(); test_split(); test_utils()
    print("all tests passed")
