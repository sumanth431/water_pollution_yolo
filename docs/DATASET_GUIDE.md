# Dataset guide

Target: **500-1500+ labelled instances per class**, varied lighting, water colour, camera height.

| Class | Where to look (free) | Search terms |
|---|---|---|
| plastic | Roboflow Universe, Kaggle, FloW (floating waste), TrashCan (underwater) | "floating plastic", "water trash", "plastic bottle river" |
| oil_spill | Roboflow Universe, Kaggle, research datasets on SAR/aerial oil spill | "oil spill", "oil slick" |
| micro_algae | Roboflow Universe, Kaggle | "algae bloom", "algae water", "cyanobacteria bloom" |

> Dataset names/availability change, so search these sites yourself and read each licence.
> "Micro algae" can mean microscope images (cells) or visible surface blooms. Surface blooms
> (camera/drone) match your abstract's drone/camera input - choose and state it in your report.

## Workflow
1. Download each dataset in **YOLOv8 format** into `data/raw/<name>/`.
2. Open each `data.yaml` to see the class names, then write `data/merge_config.json` mapping them.
3. `python scripts/merge_datasets.py --config data/merge_config.json`
4. `python scripts/check_dataset.py --data data/merged/data.yaml` - fix imbalance and bad labels.
5. Add 5-10 % background images (clean water, glare, waves, leaves) with **empty** label files.
6. Take some phone/drone photos yourself - it strengthens your project evaluation.

## Labelling your own images
Use CVAT, LabelImg or Roboflow. Use polygons for oil/algae if you plan to train segmentation.
Then `python scripts/split_dataset.py --images raw/images --labels raw/labels --out data/merged`.
