# football_broadcast_trajectories.py

import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from torchvision import transforms, models
from PIL import Image
from collections import deque

# -----------------------------
# CONFIG
# -----------------------------
IMG_W, IMG_H = 1200, 600
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIELD_TEMPLATE = {
    'yard_lines': np.linspace(0.05,0.95,10),
    'hash_left':0.1, 'hash_right':0.9,
    'endzones':[0.05,0.95]
}
ACCUMULATION_FRAMES = 10   # For stable homography
ALPHA = 0.3                 # EMA smoothing for player coordinates

# -----------------------------
# LOAD MODELS
# -----------------------------
YOLO_CLASSES = [
    'yard_5','yard_10','yard_15','yard_20','yard_25','yard_30','yard_35','yard_40','yard_45','yard_50',
    'hash_left','hash_right',
    'num_10','num_20','num_30','num_40','num_50',
    'endzone_left','endzone_right','midfield_logo','other'
]
YOLO_WEIGHTS = "runs/train/exp/weights/best.pt"
SEG_MODEL_PATH = "segmentation_model.pth"

yolo_model = YOLO(YOLO_WEIGHTS)

seg_model = models.segmentation.deeplabv3_resnet50(pretrained=False)
seg_model.classifier[-1] = torch.nn.Conv2d(256,1,1)
seg_model.load_state_dict(torch.load(SEG_MODEL_PATH,map_location=DEVICE))
seg_model.eval()
seg_model.to(DEVICE)
seg_transform = transforms.Compose([transforms.Resize((IMG_H,IMG_W)), transforms.ToTensor()])

# -----------------------------
# ACCUMULATION BUFFER
# -----------------------------
landmark_buffer = deque(maxlen=ACCUMULATION_FRAMES)

# -----------------------------
# PLAYER TRACKING STATE
# -----------------------------
player_positions_prev = {}
next_player_id = 0
trajectory_log = []  # list of dicts for CSV

# -----------------------------
# PROCESS FRAME FUNCTION
# -----------------------------
def process_frame(frame, frame_idx):
    global landmark_buffer, player_positions_prev, next_player_id, trajectory_log

    h,w,_ = frame.shape

    # --- Segmentation ---
    img_tensor = seg_transform(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        seg_out = seg_model(img_tensor)['out']
    mask = torch.sigmoid(seg_out[0,0]).cpu().numpy()
    mask = (mask>0.5).astype(np.uint8) * 255

    # --- YOLO Detection ---
    yolo_results = yolo_model(frame)[0]
    detections = []
    landmarks = []
    player_centers = []

    PLAYER_CLASS_ID = YOLO_CLASSES.index('other')  # placeholder

    for box in yolo_results.boxes:
        cls_id = int(box.cls.cpu().numpy())
        x1,y1,x2,y2 = box.xyxy.cpu().numpy()[0]
        xc = (x1+x2)/2 / w
        yc = (y1+y2)/2 / h
        class_name = YOLO_CLASSES[cls_id]

        # Overlay vs on-field
        if class_name.startswith('num_'):
            if yc<0.2: position="top_overlay"
            elif yc>0.8: position="bottom_overlay"
            elif xc<0.5: position="left_field"
            else: position="right_field"
        else:
            position="field"

        detections.append({'class':class_name,'bbox':(x1,y1,x2,y2),'norm_center':(xc,yc),'position':position})

        # Landmarks for homography
        if class_name.startswith('yard_') or class_name in ['hash_left','hash_right']:
            landmarks.append([(x1+x2)/2,(y1+y2)/2])

        # Player centers
        if cls_id==PLAYER_CLASS_ID:
            player_centers.append(np.array([(x1+x2)/2,(y1+y2)/2],dtype=np.float32))

    # --- Accumulate landmarks ---
    if landmarks:
        landmark_buffer.append(landmarks)
    accumulated_pts = [pt for pts in landmark_buffer for pt in pts]

    # --- Homography ---
    src_pts = np.array(accumulated_pts,dtype=np.float32)
    dst_pts = []
    for pt in accumulated_pts:
        y_norm = pt[1]/IMG_H
        idx = np.argmin(np.abs(FIELD_TEMPLATE['yard_lines']-y_norm))
        dst_pts.append([IMG_W/2, FIELD_TEMPLATE['yard_lines'][idx]*IMG_H])
    dst_pts = np.array(dst_pts,dtype=np.float32)

    if len(src_pts)>=4:
        H,_ = cv2.findHomography(src_pts,dst_pts,cv2.RANSAC)
    else:
        H = np.eye(3)

    # --- Player Tracking + Smoothing ---
    player_positions_curr = {}
    for center in player_centers:
        center_reshaped = np.array([[center]],dtype=np.float32)
        world_coord = cv2.perspectiveTransform(center_reshaped,H)[0,0]

        # Assign ID via nearest neighbor
        assigned = False
        for pid, prev in player_positions_prev.items():
            if np.linalg.norm(prev - world_coord)<50:
                player_positions_curr[pid] = ALPHA*world_coord + (1-ALPHA)*prev
                assigned = True
                break
        if not assigned:
            player_positions_curr[next_player_id] = world_coord
            next_player_id+=1

    player_positions_prev = player_positions_curr

    # --- Log Trajectories ---
    for pid,pos in player_positions_curr.items():
        trajectory_log.append({
            'frame': frame_idx,
            'player_id': pid,
            'x': pos[0]/IMG_W,  # normalized 0-1
            'y': pos[1]/IMG_H
        })

    # --- Visualization ---
    vis = frame.copy()
    vis[mask>0] = (0,255,0)
    for det in detections:
        x1,y1,x2,y2 = map(int,det['bbox'])
        cv2.rectangle(vis,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(vis,f"{det['class']}|{det['position']}",(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1)
    for pid,pos in player_positions_curr.items():
        cv2.circle(vis,tuple(pos.astype(int)),5,(0,0,255),-1)
        cv2.putText(vis,f"P{pid}",tuple(pos.astype(int)+np.array([5,5])),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)

    return vis

# -----------------------------
# PROCESS VIDEO
# -----------------------------
cap = cv2.VideoCapture("broadcast_football.mp4")
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    vis = process_frame(frame, frame_idx)
    cv2.imshow("Football Pipeline with Trajectories", vis)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# SAVE TRAJECTORIES
# -----------------------------
df = pd.DataFrame(trajectory_log)
df.to_csv("player_trajectories.csv",index=False)
print("Player trajectories saved to player_trajectories.csv")
