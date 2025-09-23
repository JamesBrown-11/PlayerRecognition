# football_broadcast_training.py

import os, glob, random, shutil
import cv2, numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = "football_dataset"
IMG_W, IMG_H = 1200, 600
NUM_SYNTHETIC = 500
VAL_SPLIT = 0.2
BATCH_SIZE = 4
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# YOLO Classes
YOLO_CLASSES = [
    'yard_5','yard_10','yard_15','yard_20','yard_25','yard_30','yard_35','yard_40','yard_45','yard_50',
    'hash_left','hash_right',
    'num_10','num_20','num_30','num_40','num_50',
    'endzone_left','endzone_right','midfield_logo','other'
]

import cv2
import numpy as np
import os
import random


# --------------------------
# Synthetic Frame Generator
# --------------------------
def generate_synthetic_frame(frame_width=1280, frame_height=720, yard_lines=None,
                             hash_marks=True, end_zones=True, midfield_logo=True,
                             apply_perspective=True, apply_lighting=True, apply_occlusion=True):
    """
    Generates a synthetic football field frame and its segmentation mask.

    Returns:
    - frame: RGB synthetic field image
    - mask: segmentation mask (each class as a unique pixel value)
    - bboxes: list of bounding boxes in YOLO format [class_id, x_center, y_center, w, h]
    """
    if yard_lines is None:
        yard_lines = [f"num_{i}" for i in range(0, 101, 5)]

    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    frame[:] = (0, 128, 0)  # green field background

    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = frame_height / 720
    thickness = 2
    number_offset_y = int(frame_height * 0.05)

    bboxes = []

    # Class IDs
    CLASS_YARD_LINE = 0
    CLASS_HASH = 1
    CLASS_ENDZONE = 2
    CLASS_MIDLOGO = 3
    CLASS_YARD_NUMBER = 4

    # Draw yard lines and numbers
    for num_str in yard_lines:
        try:
            num = int(num_str.split("_")[1])
            x = int(frame_width * (num / 100))

            # Yard line
            cv2.line(frame, (x, 0), (x, frame_height), (255, 255, 255), 2)
            cv2.line(mask, (x, 0), (x, frame_height), CLASS_YARD_LINE, 2)

            # Yard numbers top and bottom
            text = str(num)
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            top_y = number_offset_y + text_height
            bottom_y = frame_height - number_offset_y

            cv2.putText(frame, text, (x - text_width // 2, top_y), font, font_scale, (255, 255, 255), thickness)
            cv2.putText(frame, text, (x - text_width // 2, bottom_y), font, font_scale, (255, 255, 255), thickness)

            # Add bounding box for yard number (YOLO format)
            bbox_w = text_width
            bbox_h = text_height
            x_center = x / frame_width
            y_center_top = top_y / frame_height
            y_center_bottom = bottom_y / frame_height
            bboxes.append([CLASS_YARD_NUMBER, x_center, y_center_top, bbox_w / frame_width, bbox_h / frame_height])
            bboxes.append([CLASS_YARD_NUMBER, x_center, y_center_bottom, bbox_w / frame_width, bbox_h / frame_height])

            # Hash marks
            if hash_marks and num != 0 and num != 100:
                hash_y_top = int(frame_height * 0.15)
                hash_y_bottom = int(frame_height * 0.85)
                for side in [-1, 1]:
                    hx = x + side * int(frame_width * 0.01)
                    cv2.line(frame, (hx, hash_y_top), (hx, hash_y_top + 5), (255, 255, 255), 1)
                    cv2.line(frame, (hx, hash_y_bottom - 5), (hx, hash_y_bottom), (255, 255, 255), 1)
                    mask[hash_y_top:hash_y_top + 5, hx - 1:hx + 1] = CLASS_HASH
                    mask[hash_y_bottom - 5:hash_y_bottom, hx - 1:hx + 1] = CLASS_HASH
        except:
            continue

    # End zones
    if end_zones:
        end_zone_height = int(frame_height * 0.15)
        cv2.rectangle(frame, (0, 0), (frame_width, end_zone_height), (0, 0, 255), -1)
        cv2.rectangle(frame, (0, frame_height - end_zone_height), (frame_width, frame_height), (0, 0, 255), -1)
        mask[0:end_zone_height, :] = CLASS_ENDZONE
        mask[frame_height - end_zone_height:, :] = CLASS_ENDZONE
        # Add bounding box
        bboxes.append([CLASS_ENDZONE, 0.5, end_zone_height / (2 * frame_height), 1.0, end_zone_height / frame_height])
        bboxes.append(
            [CLASS_ENDZONE, 0.5, 1 - end_zone_height / (2 * frame_height), 1.0, end_zone_height / frame_height])

    # Midfield logo
    if midfield_logo:
        logo_radius = int(frame_height * 0.05)
        center = (frame_width // 2, frame_height // 2)
        cv2.circle(frame, center, logo_radius, (255, 255, 255), -1)
        cv2.putText(frame, "LOGO", (center[0] - logo_radius + 5, center[1] + 5), font, font_scale * 0.8, (0, 0, 0),
                    thickness)
        cv2.circle(mask, center, logo_radius, CLASS_MIDLOGO, -1)
        # Add bounding box
        bboxes.append([CLASS_MIDLOGO, center[0] / frame_width, center[1] / frame_height,
                       (2 * logo_radius) / frame_width, (2 * logo_radius) / frame_height])

    # Perspective transform
    max_shift = int(frame_width * 0.05)
    pts1 = np.float32([[0, 0], [frame_width, 0], [0, frame_height], [frame_width, frame_height]])
    pts2 = np.float32([[random.randint(0, max_shift), random.randint(0, max_shift)],
                       [frame_width - random.randint(0, max_shift), random.randint(0, max_shift)],
                       [random.randint(0, max_shift), frame_height - random.randint(0, max_shift)],
                       [frame_width - random.randint(0, max_shift), frame_height - random.randint(0, max_shift)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    frame = cv2.warpPerspective(frame, M, (frame_width, frame_height))
    mask = cv2.warpPerspective(mask, M, (frame_width, frame_height))

    # Lighting
    alpha = random.uniform(0.7, 1.3)
    beta = random.randint(-30, 30)
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Random occlusion
    for _ in range(random.randint(0, 5)):
        w, h = random.randint(20, 50), random.randint(50, 150)
        x0, y0 = random.randint(0, frame_width - w), random.randint(0, frame_height - h)
        color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
        frame[y0:y0 + h, x0:x0 + w] = color

    return frame, mask, bboxes


# --------------------------
# Generate Dataset
# --------------------------
output_dir = "synthetic_dataset"
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)

num_images = 100  # Number of images to generate

for i in range(num_images):
    frame, mask, bboxes = generate_synthetic_frame()

    # Save image
    img_path = os.path.join(output_dir, "images", f"frame_{i:04d}.png")
    cv2.imwrite(img_path, frame)

    # Save mask
    mask_path = os.path.join(output_dir, "masks", f"mask_{i:04d}.png")
    cv2.imwrite(mask_path, mask)

    # Save YOLO-format labels
    label_path = os.path.join(output_dir, "labels", f"frame_{i:04d}.txt")
    with open(label_path, 'w') as f:
        for bbox in bboxes:
            class_id, x_center, y_center, w, h = bbox
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

print(f"Generated {num_images} images, masks, and YOLO labels in '{output_dir}'")


# -----------------------
# SPLIT TRAIN/VAL
# -----------------------
def split_train_val(folder_type):
    all_files = sorted(os.listdir(os.path.join(DATA_DIR, folder_type)))
    random.shuffle(all_files)
    split_idx = int(len(all_files)*(1-VAL_SPLIT))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    for f in val_files:
        src = os.path.join(DATA_DIR, folder_type, f)
        dst = os.path.join(DATA_DIR, folder_type.replace("train","val"), f)
        shutil.move(src, dst)

split_train_val("images/train")
split_train_val("labels/train")
split_train_val("masks/train")

# -----------------------
# SEGMENTATION DATASET
# -----------------------
class FootballSegDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
        self.masks = sorted(glob.glob(os.path.join(masks_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
            mask = transforms.ToTensor()(mask)
        else:
            img = transforms.ToTensor()(img)
            mask = transforms.ToTensor()(mask)
        mask = (mask>0.5).float()
        return img, mask

seg_transform = transforms.Compose([transforms.Resize((IMG_H, IMG_W)), transforms.ToTensor()])
train_dataset = FootballSegDataset(os.path.join(DATA_DIR,"images/train"),
                                   os.path.join(DATA_DIR,"masks/train"),
                                   transform=seg_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------
# SEGMENTATION TRAINING
# -----------------------
print("Training segmentation model...")
seg_model = models.segmentation.deeplabv3_resnet50(pretrained=True)
seg_model.classifier[-1] = nn.Conv2d(256,1,1)
seg_model = seg_model.to(DEVICE)
optimizer = optim.Adam(seg_model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(NUM_EPOCHS):
    seg_model.train()
    total_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = seg_model(imgs)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {total_loss/len(train_loader):.4f}")

torch.save(seg_model.state_dict(), "segmentation_model.pth")
print("Segmentation model saved.")

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
