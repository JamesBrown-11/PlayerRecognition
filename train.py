from ultralytics import YOLO
import os
import torch

print(torch.cuda_version)
torch.cuda.set_device(0)

model = YOLO('yolov9c.yaml')

if __name__ == '__main__':

    results = model.train(data='data/data.yaml', epochs=100, imgsz=640)

    model.save()