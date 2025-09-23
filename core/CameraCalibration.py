import cv2
import numpy as np
import glob
import os
import sys

# Known real-world coordinates (soccer field example, in meters)
object_points_template = np.array([
    [0, 0, 0], [0, -53, 0], [5, 0, 0], [5, -53, 0],[10,0,0],[10,-53,0],[15,0,0],[15,-53,0],[20,0,0],[20,-53,0],[25,0,0],
    [25,-53,0],[30,0,0],[30,-53,0]
], dtype=np.float32)

objpoints = []  # 3D points
imgpoints = []  # 2D points

image_dir = "..\\data\\field\\train\\images\\*.jpg"
images = glob.glob(image_dir)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clicked_points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_points.append([x, y])
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select Points", img)

    cv2.imshow("Select Points", img)
    cv2.setMouseCallback("Select Points", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(clicked_points) != len(object_points_template):
        print(f"⚠ Skipping {fname}: need {len(object_points_template)} points, got {len(clicked_points)}")
        continue

    objpoints.append(object_points_template)
    imgpoints.append(np.array(clicked_points, dtype=np.float32))

    user_input = input("Continue 0/1")

    if user_input == "1":
        break

# Run calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("✅ Calibration complete")
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist.ravel())

# Save intrinsics
np.savez("camera_calibration.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# Test undistortion
test_img = cv2.imread(images[0])
h, w = test_img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
undistorted = cv2.undistort(test_img, mtx, dist, None, newcameramtx)
cv2.imwrite("undistorted_example.jpg", undistorted)
