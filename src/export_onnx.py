from ultralytics import YOLO

model = YOLO("runs/military/exp_finetune4/weights/best.pt")
model.export(format="onnx")
