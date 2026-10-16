"""Model loading helpers."""
from __future__ import annotations

from pathlib import Path


def find_weights(root: str = ".") -> list:
    """All trained weight files under weights/ and runs/."""
    r = Path(root)
    found = list((r / "weights").glob("*.pt")) + list(r.glob("runs/**/weights/best.pt"))
    return sorted(str(p) for p in found)


def load_model(path: str):
    from ultralytics import YOLO
    return YOLO(path)
