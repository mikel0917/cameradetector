import os
import time

import cv2
from ultralytics import YOLO

os.makedirs("snapshots", exist_ok=True)

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)

last_seen = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)
    annotated = results[0].plot()

    detections = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "label": label,
            "confidence": round(confidence, 2),
            "box": (x1, y1, x2, y2),
        })

    current = {d["label"] for d in detections if d["confidence"] > 0.5}
    if current != last_seen:
        print(f"Now seeing: {current}")
        last_seen = current

    cv2.imshow("Object Detector", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("s"):
        timestamp = int(time.time())
        filename = f"snapshots/snapshot_{timestamp}.jpg"
        cv2.imwrite(filename, annotated)
        print(f"Saved {filename} - objects: {current}")

cap.release()
cv2.destroyAllWindows()
