from ultralytics import YOLO
import numpy as np


def load_yolo_model(model_path="models/yolov8n.pt"):
    return YOLO(model_path)


def run_detection(model, image):
    results = model(image, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        detections.append({"label": label, "box": (x1, y1, x2, y2), "center": (cx, cy)})
    return detections
