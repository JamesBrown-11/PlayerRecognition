from ultralytics import YOLO
import cv2
import numpy as np
import os
from core.LineIdentification import LineIdentification

player_model = YOLO('runs/detect/train/weights/best.pt')
#marking_model = YOLO('runs/detect/train8/weights/best.pt')

video_path = 'film/2023 Wk1 All 22 Dolphins Chargers Coaches Film.mp4'
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        player_results = player_model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = player_results[0].plot()

        # marking_results = marking_model.track(annotated_frame, persist=True)
        # annotated_frame2 = marking_results[0].plot(img=annotated_frame)

        # marking_boxes = marking_results[0].boxes.xyxy.cpu()
        # for box in marking_boxes:
        #     x1, y1, x2, y2 = np.array(box).astype(int)
        #     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        line_identifier = LineIdentification()
        line_identifier.edge_detection(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv9 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()