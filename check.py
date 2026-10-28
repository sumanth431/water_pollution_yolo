import sys
from ultralytics import YOLO

images = sys.argv[1:]
if not images:
    sys.exit("usage: python check.py image1.jpg image2.jpg ...")

models = {"plastic": YOLO("weights/plastic.pt"), "oil+algae": YOLO("weights/oil_algae.pt")}
for img in images:
    print("\n", img)
    for name, m in models.items():
        r = m.predict(img, conf=0.05, verbose=False)[0]
        found = [(m.names[int(c)], round(float(p), 2)) for c, p in zip(r.boxes.cls, r.boxes.conf)]
        print(f"  {name:10s} -> {found if found else 'nothing'}")
        