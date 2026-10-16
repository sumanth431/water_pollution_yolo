"""Evaluate a trained model on the test split and save metrics to results/.

  python evaluate.py --weights runs/yolov9c_water/weights/best.pt
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def evaluate(weights, data, imgsz=640, split="test"):
    model = YOLO(weights)
    m = model.val(data=data, split=split, imgsz=imgsz)
    per_class = {model.names[int(i)]: float(m.box.maps[int(i)]) for i in m.box.ap_class_index}
    return {
        "weights": weights,
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "map50": float(m.box.map50),
        "map50_95": float(m.box.map),
        "per_class_map50_95": per_class,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    res = evaluate(a.weights, a.data, a.imgsz, a.split)
    Path("results").mkdir(exist_ok=True)
    out = Path("results") / f"metrics_{Path(a.weights).parent.parent.name}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("Saved:", out)


if __name__ == "__main__":
    main()
