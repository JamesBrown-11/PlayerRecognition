# football_broadcast_training.py

import os, random, shutil
import cv2, numpy as np
import torch

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
        yard_lines = [f"num_{i}" for i in range(5, 51, 5)]

    for i in range(45, 0, -5):
        yard_lines.append(f"num_{i}")

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
    CLASS_SIDELINE = 5  # new class id

    # Draw sidelines (top and bottom horizontal lines)
    cv2.line(frame, (0, 0), (frame_width, 0), (255, 255, 255), 20)  # top sideline
    cv2.line(frame, (0, frame_height - 1), (frame_width, frame_height - 1), (255, 255, 255), 20)  # bottom sideline

    # Add to segmentation mask
    mask[0:4, :] = CLASS_SIDELINE  # top
    mask[frame_height - 4:frame_height, :] = CLASS_SIDELINE  # bottom

    # YOLO bboxes (optional, if you want them as objects)
    # bboxes.append([CLASS_SIDELINE, 0.5, (2 / frame_height), 1.0, (4 / frame_height)])  # top line
    # bboxes.append([CLASS_SIDELINE, 0.5, (frame_height - 2) / frame_height, 1.0, (4 / frame_height)])  # bottom line

    # Draw yard lines and numbers
    flip = False
    increase = 1
    for num_str in yard_lines:
        try:
            num = int(num_str.split("_")[1])
            available_field_area = frame_width * 0.8
            if flip:
                x = int(available_field_area * ((num + increase * 10) / 100) + frame_width * 0.1)
                increase += 1
            else:
                x = int(available_field_area * (num / 100) + frame_width * 0.1)

            if num == 50:
                flip = True



            # Yard line
            cv2.line(frame, (x, 0), (x, frame_height), (255, 255, 255), 2)
            cv2.line(mask, (x, 0), (x, frame_height), CLASS_YARD_LINE, 2)

            if num % 10 == 0:
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
        end_zone_width = int(frame_width * 0.1)  # 10% of width
        # Left end zone
        cv2.rectangle(frame, (0, 0), (end_zone_width, frame_height), (255, 255, 255), 20)
        mask[:, 0:end_zone_width] = CLASS_ENDZONE
        bboxes.append([CLASS_ENDZONE, (end_zone_width / 2) / frame_width, 0.5,
                       end_zone_width / frame_width, 1.0])
        # Right end zone
        cv2.rectangle(frame, (frame_width - end_zone_width, 0), (frame_width, frame_height), (255, 255, 255), 20)
        mask[:, frame_width - end_zone_width:] = CLASS_ENDZONE
        bboxes.append([CLASS_ENDZONE, (frame_width - end_zone_width / 2) / frame_width, 0.5,
                       end_zone_width / frame_width, 1.0])

    # Midfield logo
    # if midfield_logo:
    #     logo_radius = int(frame_height * 0.05)
    #     center = (frame_width // 2, frame_height // 2)
    #     cv2.circle(frame, center, logo_radius, (255, 255, 255), -1)
    #     cv2.putText(frame, "LOGO", (center[0] - logo_radius + 5, center[1] + 5), font, font_scale * 0.8, (0, 0, 0),
    #                 thickness)
    #     cv2.circle(mask, center, logo_radius, CLASS_MIDLOGO, -1)
    #     # Add bounding box
    #     bboxes.append([CLASS_MIDLOGO, center[0] / frame_width, center[1] / frame_height,
    #                    (2 * logo_radius) / frame_width, (2 * logo_radius) / frame_height])

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



# -----------------------
# DIRECTORY SETUP
# -----------------------
def init_directories():
    folders = [
        "images/train", "images/val",
        "labels/train", "labels/val",
        "masks/train", "masks/val"
    ]
    for folder in folders:
        os.makedirs(os.path.join(DATA_DIR, folder), exist_ok=True)


if __name__ == "__main__":
    init_directories()

    for i in range(NUM_SYNTHETIC):
        frame, mask, bboxes = generate_synthetic_frame()

        # Save image
        img_path = os.path.join(DATA_DIR, "images/train", f"frame_{i:04d}.png")
        cv2.imwrite(img_path, frame)

        # Save mask
        mask_path = os.path.join(DATA_DIR, "masks/train", f"mask_{i:04d}.png")
        cv2.imwrite(mask_path, mask)

        # Save YOLO-format labels
        label_path = os.path.join(DATA_DIR, "labels/train", f"frame_{i:04d}.txt")
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                class_id, x_center, y_center, w, h = bbox
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

    print(f"Generated {NUM_SYNTHETIC} images, masks, and YOLO labels in '{DATA_DIR}'")


    split_train_val("images/train")
    split_train_val("labels/train")
    split_train_val("masks/train")