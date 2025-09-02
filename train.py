from ultralytics import YOLO
import os
import torch

if __name__ == '__main__':
    print(torch.version.cuda)
    torch.cuda.set_device(0)

    model = YOLO('yolov9c.yaml')

    results = model.train(data='data/data.yaml', epochs=100, imgsz=640)