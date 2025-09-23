# football_broadcast_training.py

import os, glob, random, shutil
import cv2, numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torchvision import models
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "football_dataset"
IMG_W, IMG_H = 1280, 720
IMG_SIZE = 640
NUM_SYNTHETIC = 500
VAL_SPLIT = 0.2
BATCH_SIZE = 2
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 6  # background + 5 field markings

# YOLO Classes
YOLO_CLASSES = [
    'yard_5','yard_10','yard_15','yard_20','yard_25','yard_30','yard_35','yard_40','yard_45','yard_50',
    'hash_left','hash_right',
    'num_10','num_20','num_30','num_40','num_50',
    'endzone_left','endzone_right','midfield_logo','other'
]



# -----------------------
# SEGMENTATION DATASET
# -----------------------
class FootballSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_size=640):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")) +
                                  glob.glob(os.path.join(image_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.img_size = img_size

        assert len(self.image_paths) == len(self.mask_paths), \
            f"Number of images ({len(self.image_paths)}) and masks ({len(self.mask_paths)}) must match!"

        # Image transforms
        self.img_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        # Mask transform
        self.mask_resize = T.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load mask (grayscale)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # Apply transforms
        img = self.img_transform(img)  # [3, H, W]

        mask = T.ToPILImage()(mask)
        mask = self.mask_resize(mask)
        mask = T.ToTensor()(mask).squeeze(0).long()  # [H, W], int64

        return img, mask

# Dataset & Loader
train_dataset = FootballSegDataset(os.path.join(DATA_DIR, "images/train"),
                                   os.path.join(DATA_DIR, "masks/train"),
                                   IMG_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model (DeepLabV3)
seg_model = models.segmentation.deeplabv3_resnet50(pretrained=True)
seg_model.classifier[-1] = nn.Conv2d(256, NUM_CLASSES, 1)  # multi-class output
seg_model = seg_model.to(DEVICE)

optimizer = optim.Adam(seg_model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()  # expects class indices

# Training loop
print("Training segmentation model...")
for epoch in range(NUM_EPOCHS):
    seg_model.train()
    total_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)  # masks: [B, H, W]

        optimizer.zero_grad()
        outputs = seg_model(imgs)['out']  # [B, num_classes, H, W]
        loss = criterion(outputs, masks)  # masks are class indices
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

# Save model
torch.save(seg_model.state_dict(), "segmentation_multiclass.pth")
print("✅ Multi-class segmentation model saved to segmentation_multiclass.pth")

# -----------------------
# YOLOv8 TRAINING
# -----------------------
print("Training YOLOv8 model...")
# Create yolo_data.yaml
yolo_yaml_path = os.path.join(DATA_DIR, "yolo_data.yaml")
with open(yolo_yaml_path,"w") as f:
    f.write(f"train: {DATA_DIR}/images/train\n")
    f.write(f"val: {DATA_DIR}/images/val\n")
    f.write(f"nc: {len(YOLO_CLASSES)}\n")
    f.write(f"names: {YOLO_CLASSES}\n")

yolo_model = YOLO("yolov8n.pt")
yolo_model.train(
    data=yolo_yaml_path,
    epochs=NUM_EPOCHS,
    imgsz=IMG_H,
    batch=BATCH_SIZE,
    device=0 if torch.cuda.is_available() else -1
)
print("YOLOv8 training complete.")
