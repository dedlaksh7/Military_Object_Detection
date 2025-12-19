import os
import cv2
import zipfile
from ultralytics import YOLO

MODEL_PATH = "../model/best.pt"
IMAGE_DIR = "../test_images"
OUTPUT_DIR = "../result"
ZIP_PATH = "../result/results.zip"

CONF_THRESHOLD = 0.25

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH, task="detect")

txt_files = []

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    results = model(img, conf=CONF_THRESHOLD)

    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(OUTPUT_DIR, txt_name)

    with open(txt_path, "w") as f:
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0]
                x_center = ((x1 + x2) / 2) / w
                y_center = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h

                f.write(
                    f"{cls_id} {x_center:.6f} {y_center:.6f} "
                    f"{bw:.6f} {bh:.6f} {conf:.6f}\n"
                )

    txt_files.append(txt_path)
    print(f"✔ Saved: {txt_name}")

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
    for file in txt_files:
        zipf.write(file, os.path.basename(file))

print("\n✅ RESULTS READY")
print(f"📦 ZIP FILE: {ZIP_PATH}")
