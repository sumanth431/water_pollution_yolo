# Detection of Plastic, Oil Spills and Micro Algae (YOLOv9)

Batch 24 · CSE (Data Science) · Santhiram Engineering College, Nandyal
Team: P. Nagadharani, T. Jayanthi, Y. Sreevani · Guide: Mr. B. Venkatacharankumar, M.Tech

A deep-learning system that detects **plastic waste, oil spills and micro algae** in water-body
images/video (cameras or drones), draws labelled bounding boxes, counts detections and raises alerts.

## Project structure
```
app.py                 Streamlit web app (image + video upload, alerts)
train.py               Train YOLOv9 / YOLOv8 / YOLO11 (detect or -seg)
evaluate.py            Precision, Recall, mAP on the test split -> results/*.json
predict.py             CLI inference on image / folder / video / webcam
compare_models.py      Train+evaluate several models -> results/model_comparison.csv
data/data.yaml         Dataset config (3 classes)
scripts/
  merge_datasets.py    Merge oil / algae / plastic datasets with unified class IDs
  split_dataset.py     70/20/10 split for your own labelled images
  check_dataset.py     Class balance + label sanity check
src/                   utils (alerts, counts, coverage), model loader
notebooks/             Colab notebook for free-GPU training
tests/                 Tests for dataset tools and helpers
docs/                  Report outline, viva Q&A
```

## Quick start

**1. Install** (Python 3.9+)
```bash
pip install -r requirements.txt
```

**2. Get data** (this is the hard part - see docs/DATASET_GUIDE.md)
Download one dataset per class from Roboflow Universe / Kaggle in *YOLOv8 format*, then:
```bash
cp data/merge_config.example.json data/merge_config.json   # edit paths and class maps
python scripts/merge_datasets.py --config data/merge_config.json
python scripts/check_dataset.py --data data/merged/data.yaml
```
Update `path:` in `data/data.yaml` (or just use `data/merged/data.yaml`).

**3. Train** - use Colab/Kaggle free GPU (open `notebooks/water_pollution_yolov9_colab.ipynb`) or locally with a GPU:
```bash
python train.py --model yolov9t.pt --epochs 3 --name smoke_test      # 5-minute pipeline test
python train.py --model yolov9c.pt --epochs 100 --name yolov9c_water # real run
```
If out of memory: `--batch 8` or use `yolov9s.pt`.

**4. Evaluate**
```bash
python evaluate.py --weights runs/yolov9c_water/weights/best.pt
```
Curves and the confusion matrix are in `runs/yolov9c_water/` - use them in the report.

**5. Compare with baselines** (for the "results" chapter)
```bash
python compare_models.py --models yolov9c.pt yolov8s.pt yolo11s.pt --epochs 100
```

**6. Run the app**
```bash
cp runs/yolov9c_water/weights/best.pt weights/best.pt
streamlit run app.py
```
Webcam / drone stream: `python predict.py --weights weights/best.pt --source 0 --show`

## Improving results
- Oil and algae have irregular shapes: label them with **polygons** and train `yolov9c-seg.pt`
  (`python train.py --model yolov9c-seg.pt --name seg_water`). The app already estimates coverage from masks.
- Missed small plastics: `--imgsz 960` or `1280`.
- False positives from glare/reflections: add "clean water" images with empty label files (background samples).
- Weakest class low? Add data for that class first, before tuning hyperparameters.

## Notes
- Ultralytics is AGPL-3.0: fine for academic work; check licensing before commercial use.
- Check each dataset's licence before publishing results or the model.
- Model weights are not included - you train them on your data.
