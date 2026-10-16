"""Run detection on an image, folder, video or webcam.

  python predict.py --weights weights/best.pt --source test.jpg
  python predict.py --weights weights/best.pt --source video.mp4
  python predict.py --weights weights/best.pt --source 0 --show      # webcam
"""
import argparse

from src.model import load_model
from src.utils import build_alerts, counts_from_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="path, folder, URL, or 0 for webcam")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--show", action="store_true", help="open a window (local machine only)")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    source = int(a.source) if a.source.isdigit() else a.source
    model = load_model(a.weights)

    for i, r in enumerate(model.predict(source=source, conf=a.conf, iou=a.iou, imgsz=a.imgsz,
                                        stream=True, show=a.show, save=not a.no_save,
                                        project="results", name="predict", exist_ok=True)):
        counts = counts_from_result(r, model.names)
        for level, msg in build_alerts(counts):
            print(f"[{i}] {level.upper()}: {msg}")


if __name__ == "__main__":
    main()
