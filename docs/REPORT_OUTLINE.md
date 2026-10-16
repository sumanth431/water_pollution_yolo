# Report outline

1. **Abstract** (tighten the current one; fix "Concept" -> "Content/Object" Detection in the title)
2. **Introduction** - water pollution, need for automated monitoring, objectives, scope
3. **Literature survey** - YOLO family (v3-v9), oil-spill detection, algae bloom detection, floating-waste detection (5-10 papers, table)
4. **Existing vs proposed system** - manual/satellite vs YOLOv9 camera/drone pipeline
5. **System design** - architecture diagram, data flow, use-case, module list
6. **Methodology**
   - Dataset (sources, class counts, split, licences)
   - Preprocessing and augmentation
   - YOLOv9 architecture (GELAN backbone, PGI) in brief
   - Training setup (epochs, batch, imgsz, optimizer, hardware)
7. **Implementation** - code structure, Streamlit app, screenshots
8. **Results and discussion** - P/R/mAP table, per-class results, confusion matrix, PR/F1 curves, sample detections, comparison with YOLOv8/YOLO11, failure cases (glare, small objects)
9. **Conclusion and future scope** - segmentation, Sentinel-2/SAR, edge deployment (Jetson/RPi), alert system, drone integration
10. **References**

## Likely viva questions
- Why YOLOv9 over Faster R-CNN or YOLOv8? (speed/accuracy trade-off, PGI/GELAN, you benchmarked it)
- What is mAP@0.5 vs mAP@0.5:0.95? What is IoU?
- How did you handle class imbalance and glare/reflections?
- Why are boxes a poor fit for oil and algae? (irregular shape -> segmentation)
- How would it run in real time on a drone? (ONNX/TensorRT, smaller model, image size)
- What are the limitations? (dataset size, domain shift, weather, night)
