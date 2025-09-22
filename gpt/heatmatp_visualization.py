# football_trajectory_visualization.py

import cv2
import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
IMG_W, IMG_H = 1200, 600
FIELD_COLOR = (0,128,0)
PLAYER_COLORS = [
    (255,0,0),(0,0,255),(0,255,255),(255,255,0),(255,0,255),(0,128,255),(128,0,128),(0,255,0)
]

# Load trajectory CSV from previous pipeline
trajectory_df = pd.read_csv("player_trajectories.csv")

# -----------------------------
# CREATE BASE FIELD TEMPLATE
# -----------------------------
field_template = np.zeros((IMG_H,IMG_W,3),dtype=np.uint8)
field_template[:] = FIELD_COLOR

# Add optional lines for yard lines (visual guide)
num_yard_lines = 10
for i in range(num_yard_lines):
    y = int(IMG_H * (0.05 + i*0.1))
    cv2.line(field_template,(0,y),(IMG_W,y),(255,255,255),1)

# -----------------------------
# TRAJECTORY VISUALIZATION
# -----------------------------
# Parameters
TRAJECTORY_LENGTH = 20  # frames to show past positions

# Prepare player position history
player_history = {}

# Process frame by frame
frame_ids = trajectory_df['frame'].unique()
for frame_idx in frame_ids:
    frame_vis = field_template.copy()
    frame_data = trajectory_df[trajectory_df['frame']==frame_idx]

    # Update player history
    for _, row in frame_data.iterrows():
        pid = row['player_id']
        x = int(row['x']*IMG_W)
        y = int(row['y']*IMG_H)
        if pid not in player_history:
            player_history[pid] = []
        player_history[pid].append((x,y))
        # Keep last TRAJECTORY_LENGTH positions
        if len(player_history[pid])>TRAJECTORY_LENGTH:
            player_history[pid] = player_history[pid][-TRAJECTORY_LENGTH:]

    # Draw trajectories
    for pid, positions in player_history.items():
        color = PLAYER_COLORS[pid % len(PLAYER_COLORS)]
        for j in range(1,len(positions)):
            cv2.line(frame_vis,positions[j-1],positions[j],color,2)
        # Draw current position
        cv2.circle(frame_vis,positions[-1],5,color,-1)
        cv2.putText(frame_vis,f"P{pid}",(positions[-1][0]+5,positions[-1][1]+5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)

    # Optional: create heatmap overlay
    heatmap = np.zeros((IMG_H,IMG_W),dtype=np.float32)
    for positions in player_history.values():
        for x,y in positions:
            heatmap[y,x] += 1
    heatmap = np.clip(heatmap/heatmap.max(),0,1)
    heatmap_color = cv2.applyColorMap((heatmap*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame_vis,0.7,heatmap_color,0.3,0)

    cv2.imshow("Player Trajectories & Heatmap", overlay)
    if cv2.waitKey(30) & 0xFF==ord('q'):
        break

cv2.destroyAllWindows()
