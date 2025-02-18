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

class LineIdentification:
    def __init__(self):
        self.img = None
        self.pixels = None


    def extract_pixels(self, img):
        self.img = Image.open(img)
        self.pixels = self.img.load()


    def calculate_pixel_luminance(self, r, g, b):
        return (r * 0.2126) + (g * 0.7152) + (b * 0.0722)

    def find_lines(self):
        size = self.img.size

        for x in range(size[0]):
            for y in range(size[1]):
                current_pixel = self.pixels[x,y]
                pixel_r = self.pixels[x, y][0]
                pixel_g = self.pixels[x, y][1]
                pixel_b = self.pixels[x, y][2]
                pixel_luminance = self.calculate_pixel_luminance(pixel_r, pixel_g, pixel_b)
                # check if the rgb values of the pixel are white
                # each value must be greater tha 240
                if pixel_luminance >= SIGMA_L:
                    # Since this is considered a white pixel, determine if the pixels LINE_WIDTH linearly
                    # Must account for if the current pixel is near the left edge
                    left_most_x = max(x - LINE_WIDTH, 0)
                    right_most_x = min(x + LINE_WIDTH, size[0]-1)
                    top_most_y = max(y - LINE_WIDTH, 0)
                    bottom_most_y = min(y + LINE_WIDTH, size[1]-1)

                    left_most_pixel = self.pixels[left_most_x, y]
                    right_most_pixel = self.pixels[right_most_x, y]
                    top_most_pixel = self.pixels[x, top_most_y]
                    bottom_most_pixel = self.pixels[x, bottom_most_y]

                    left_most_pixel_luminance = self.calculate_pixel_luminance(left_most_pixel[0], left_most_pixel[1], left_most_pixel[2])
                    right_most_pixel_luminance = self.calculate_pixel_luminance(right_most_pixel[0], right_most_pixel[1], right_most_pixel[2])
                    top_most_pixel_luminance = self.calculate_pixel_luminance(top_most_pixel[0], top_most_pixel[1], top_most_pixel[2])
                    bottom_most_pixel_luminance = self.calculate_pixel_luminance(bottom_most_pixel[0], bottom_most_pixel[1], bottom_most_pixel[2])

                    if ((pixel_luminance - left_most_pixel_luminance >= SIGMA_D and pixel_luminance - right_most_pixel_luminance >= SIGMA_D)
                            or (pixel_luminance - top_most_pixel_luminance >= SIGMA_D and pixel_luminance - bottom_most_pixel_luminance >= SIGMA_D)):
                        # print('White line detection')
                        # If this check is passed, the candidate pixel is considered on a line
                        # TODO: We must exclude white pixels that are in textrued regions i.e. letters in logos, white areas in stadium, or spectators
                        # TODO: Hough Line Detection

                        self.img.putpixel((x, y), (0, 0, 0))
        # self.img.show()

    def edge_detection(self, frame):
        size = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        blurred = cv2.GaussianBlur(src=gray, ksize=(3,5), sigmaX=0.5)

        edges = cv2.Canny(blurred, 70, 135)

        cv2.imshow("edges", edges)
        # cv2.waitKey(0)

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
