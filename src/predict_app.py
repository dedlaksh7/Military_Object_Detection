from ultralytics import YOLO
import cv2
import sys

# Load model
model = YOLO("best.pt")

# Read input image path
image_path = sys.argv[1]

# Run prediction
results = model.predict(image_path, imgsz=640, conf=0.25, device="cpu")

# Load image
img = cv2.imread(image_path)

# Draw detections
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        confidence = float(box.conf[0])
        label = f"{model.names[cls]} {confidence:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

# ---------- FIX WINDOW SIZE ----------
h, w = img.shape[:2]
max_height = 800  # the window will shrink to this height
scale = max_height / h
resized = cv2.resize(img, (int(w * scale), int(h * scale)))

cv2.imshow("Detections", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
