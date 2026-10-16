"""Train + evaluate several models with identical settings and save a comparison table.

  python compare_models.py --models yolov9c.pt yolov8s.pt yolo11s.pt --epochs 100
"""
import argparse
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from evaluate import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["yolov9c.pt", "yolov8s.pt", "yolo11s.pt"])
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    rows = []
    for w in a.models:
        name = Path(w).stem + "_cmp"
        YOLO(w).train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                      patience=20, project="runs", name=name, exist_ok=True)
        r = evaluate(f"runs/{name}/weights/best.pt", a.data, a.imgsz)
        rows.append({"Model": Path(w).stem, "Precision": r["precision"], "Recall": r["recall"],
                     "mAP@0.5": r["map50"], "mAP@0.5:0.95": r["map50_95"]})

    df = pd.DataFrame(rows).round(3)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/model_comparison.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
