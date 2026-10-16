"""Streamlit web app: upload an image/video and detect plastic, oil spills and micro algae.

Run:  streamlit run app.py
"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.model import find_weights, load_model
from src.utils import build_alerts, counts_from_result, coverage_from_result

st.set_page_config(page_title="Water Pollution Detection", page_icon="🌊", layout="wide")
st.title("🌊 Water Pollution Detection")
st.caption("Plastic waste • Oil spills • Micro algae — YOLOv9")

# ---------------- sidebar ----------------
st.sidebar.header("Model")
available = find_weights()
if available:
    weights = st.sidebar.selectbox("Trained weights", available)
else:
    weights = st.sidebar.text_input("Weights path", "weights/best.pt")
uploaded_w = st.sidebar.file_uploader("...or upload a .pt file", type=["pt"])
if uploaded_w is not None:
    tmp_w = Path(tempfile.gettempdir()) / uploaded_w.name
    tmp_w.write_bytes(uploaded_w.getbuffer())
    weights = str(tmp_w)

conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, 0.40, 0.05)
iou = st.sidebar.slider("IoU (NMS) threshold", 0.1, 0.9, 0.50, 0.05)
imgsz = st.sidebar.select_slider("Inference size", [320, 480, 640, 960, 1280], value=640)
stride = st.sidebar.slider("Video: process every Nth frame", 1, 10, 2)


@st.cache_resource(show_spinner="Loading model...")
def get_model(path: str):
    return load_model(path)


if not Path(weights).exists():
    st.warning(f"Weights not found: `{weights}`. Train a model first (see README) and place "
               "`best.pt` in the `weights/` folder, or upload it in the sidebar.")
    st.stop()

model = get_model(weights)
names = model.names


def show_report(counts, coverage):
    for level, msg in build_alerts(counts):
        getattr(st, level)(msg)
    if counts:
        df = pd.DataFrame({"Class": list(counts), "Detections": list(counts.values()),
                           "Approx. frame coverage (%)": [round(coverage.get(k, 0), 2) for k in counts]})
        st.dataframe(df, hide_index=True, use_container_width=True)


tab_img, tab_vid = st.tabs(["🖼️ Image", "🎥 Video"])

# ---------------- image ----------------
with tab_img:
    f = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="img")
    if f:
        img = np.array(Image.open(f).convert("RGB"))
        r = model.predict(img[:, :, ::-1], conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        annotated = r.plot()[:, :, ::-1]
        c1, c2 = st.columns(2)
        c1.image(img, caption="Input", use_container_width=True)
        c2.image(annotated, caption="Detections", use_container_width=True)
        show_report(counts_from_result(r, names), coverage_from_result(r, names))

# ---------------- video ----------------
with tab_vid:
    v = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    if v and st.button("Run detection on video", type="primary"):
        src = Path(tempfile.mkdtemp()) / v.name
        src.write_bytes(v.getbuffer())
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        W, H = int(cap.get(3)), int(cap.get(4))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        out_path = str(src.with_name("annotated.mp4"))
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

        view, bar = st.empty(), st.progress(0.0)
        peak, frames_with = {}, {}
        last = None
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                r = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
                last = r.plot()
                for k, n in counts_from_result(r, names).items():
                    peak[k] = max(peak.get(k, 0), n)
                    frames_with[k] = frames_with.get(k, 0) + 1
                view.image(last[:, :, ::-1], use_container_width=True)
            writer.write(last if last is not None else frame)
            i += 1
            bar.progress(min(i / total, 1.0))
        cap.release()
        writer.release()
        bar.empty()

        st.success(f"Processed {i} frames.")
        show_report(peak, {})
        if frames_with:
            st.caption("Frames (of those analysed) containing each class: " +
                       ", ".join(f"{k}: {n}" for k, n in frames_with.items()))
        with open(out_path, "rb") as fh:
            st.download_button("⬇️ Download annotated video", fh.read(), file_name="annotated.mp4", mime="video/mp4")

st.sidebar.markdown("---")
st.sidebar.caption("Live webcam / drone feed: use `python predict.py --source 0 --show` on a local machine.")
