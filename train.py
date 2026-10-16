"""Train YOLOv9 (or any Ultralytics model) on the water-pollution dataset.

Examples:
  python train.py --model yolov9c.pt --epochs 100
  python train.py --model yolov9t.pt --epochs 3 --name smoke_test
  python train.py --resume runs/yolov9c_water/weights/last.pt
"""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov9c.pt", help="yolov9t/s/m/c/e.pt, yolov8s.pt, yolo11s.pt, *-seg.pt ...")
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16, help="lower to 8/4 on out-of-memory")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="yolov9c_water")
    ap.add_argument("--device", default=None, help="0 for GPU, cpu for CPU (default: auto)")
    ap.add_argument("--resume", default=None, help="path to last.pt to resume")
    a = ap.parse_args()

    if a.resume:
        YOLO(a.resume).train(resume=True)
        return

    model = YOLO(a.model)
    model.train(
        data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        patience=a.patience, project=a.project, name=a.name, exist_ok=True,
        device=a.device, plots=True,
        # augmentation tuned for water scenes (glare, colour changes)
        mosaic=1.0, fliplr=0.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    )


if __name__ == "__main__":
    main()
