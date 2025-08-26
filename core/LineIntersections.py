import math

from PIL import Image

import cv2
import numpy as np
import time
import subprocess

LINE_WIDTH = 10
SIGMA_L = 128
SIGMA_D = 20


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # Method used to display X and Y coordinates
    # of a point
    def displayPoint(self, p):
        print(f"({p.x}, {p.y})")

class LineExtractor:
    def __init__(self):
        self.img = None
        self.pixels = None

    def edge_detection(self, frame):
        size = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        blurred = cv2.GaussianBlur(src=gray, ksize=(3,5), sigmaX=0.5)

        # cv2.imshow("blurred", blurred)

        edges = cv2.Canny(blurred, 70, 135)

        # cv2.imshow("edges", edges)
        # cv2.waitKey(0)

        return edges

    def calculate_orientation(self, edges):
        # Apply HoughLinesP method to directly obtain line end points
        lines_list = []
        quadratics_list = []
        lines = cv2.HoughLinesP(
            edges,  # Input edge image
            1,  # Distance resolution in pixels
            np.pi / 180,  # Angle resolution in radians
            threshold=200,  # Min number of votes for valid line
            minLineLength=300,  # Min allowed length of line
            maxLineGap=10  # Max allowed gap between line for joining them
        )

        num_lines_horizontal = 0
        num_lines_vertical = 0
        # Remove any lines above lowest top horizontal line (assume to be the sideline or endzone)
        # Disregard any lines below the midpoint_y
        if lines is None:
            return 'none-type'

        for points in lines:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]

            # cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            top = y2 - y1
            bottom = x2 - x1

            if bottom == 0:
                slope = 0.0
            else:
                slope = abs(top / bottom)

            if slope < 0.5:
                # Horizontal line
                num_lines_horizontal += 1
            else:
                num_lines_vertical += 1

        if num_lines_horizontal > num_lines_vertical:
            return 'endzone'
        else:
            return 'sideline'

    def draw_lines(self, frame, edges):
        # Apply HoughLinesP method to directly obtain line end points
        lines_list = []
        quadratics_list = []
        lines = cv2.HoughLines(
            edges,  # Input edge image
            1,  # Distance resolution in pixels
            np.pi / 180,  # Angle resolution in radians
            threshold=200,  # Min number of votes for valid line
        )

        # The below for loop runs till r and theta values
        # are in the range of the 2d array
        for r_theta in lines:
            arr = np.array(r_theta[0], dtype=np.float64)
            r, theta = arr
            # Stores the value of cos(theta) in a
            a = np.cos(theta)

            # Stores the value of sin(theta) in b
            b = np.sin(theta)

            # x0 stores the value rcos(theta)
            x0 = a * r

            # y0 stores the value rsin(theta)
            y0 = b * r

            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000 * (-b))

            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 1000 * (a))

            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000 * (-b))

            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 1000 * (a))

            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
            # (0,0,255) denotes the colour of the line to be
            # drawn. In this case, it is red.
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            return frame
