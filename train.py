from ultralytics import YOLO
import os

model = YOLO('yolov9c.yaml')

results = model.train(data='data/data.yaml', epochs=100, imgsz=640)

model.save()