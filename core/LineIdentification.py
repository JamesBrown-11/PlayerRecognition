import math
import sys
import time

from PIL import Image

import cv2
import numpy as np
import statistics

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

    def edge_detection(self, frame, threshold=200, minLineLength=500, maxLineGap=100):
        size = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        blurred = cv2.GaussianBlur(src=gray, ksize=(3,5), sigmaX=0.5)

        edges = cv2.Canny(blurred, 70, 135)

        # Apply HoughLinesP method to
        # to directly obtain line end points
        lines_list = []
        quadratics_list = []
        lines = cv2.HoughLinesP(
            edges,  # Input edge image
            1,  # Distance resolution in pixels
            np.pi / 180,  # Angle resolution in radians
            threshold=threshold,  # Min number of votes for valid line
            minLineLength=minLineLength,  # Min allowed length of line
            maxLineGap=maxLineGap  # Max allowed gap between line for joining them
        )

        top_threshold_y = size[0] - int(size[0] * 0.95)
        bottom_threshold_y = size[0] * 0.95

        # Iterate over points
        count = 0
        for points in lines:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]

            # if (y1 > top_threshold_y or y2 > top_threshold_y) and (y1 < bottom_threshold_y or y2 < bottom_threshold_y):
                # Draw the lines joining the points

                # Maintain a simples lookup list for points
            lines_list.append([(x1, y1), (x2, y2)])
            count += 1

        self.merge_line(lines_list, frame)
        self.draw_line(lines_list, frame)
        # cv2.imwrite(frame, image)
        # modified_image = cv2.imread('modified.png')
        # cv2.imshow('Modified Image', modified_image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # print("Done")

    def merge_line(self, line_list, frame):
        merged_lines = []
        i = 0
        while i < len(line_list):
            line = line_list[i]
            A = Point(line[0][0], line[0][1] * -1)
            B = Point(line[1][0], line[1][1] * -1)

            slope, y_intercept = self.calculate_slope_y_intercept(A, B)

            intersecting_lines = []
            copied_frame = frame.copy()
            j = 0
            while j < len(line_list)-1:
                if j != i:
                    found_intersecting_line = False

                    other_line = line_list[j]
                    C = Point(other_line[0][0], other_line[0][1] * -1)
                    D = Point(other_line[1][0], other_line[1][1] * -1)

                    other_slope, other_y_intercept = self.calculate_slope_y_intercept(C, D)

                    # Vertical line check:
                    if slope is None:
                        if other_slope is None:
                            # If both slopes are None, both lines are vertical
                            # Check if the distance between them is less than 50
                            distance = abs(A.x - C.x)

                            if distance < 50:
                                print ("Line {} and {} are both vertical and should merge".format(i, j))
                        else:
                            # The first line is vertical but the second line is not
                            # Need to determine of the lines intersect on the segment in view
                            # Where does the second line intersect with the first line
                            intersection_y = other_slope * A.x + other_y_intercept

                            if A.y > B.y:
                                top_y = A.y
                                bottom_y = B.y
                            else:
                                top_y = B.y
                                bottom_y = A.y

                            # Check if y is greater than A.y or less than B.y
                            if top_y >= intersection_y >= bottom_y:
                                # Check if the second line slope is 50 or greater
                                # If it is, these lines should merge
                                if abs(other_slope) >= 50:
                                    print("Line {} is vertical and line {} has a slope greater than 50, they should merge".format(i, j))

                    # Horizontal line check
                    elif slope == 0:
                        if other_slope == 0:
                            # Both these lines are horizontal
                            distance = abs(A.y - C.y)

                            if distance < 50:
                                print ("Line {} and {} are both horizontal and should merge".format(i, j))
                        else:
                            # The first line is horizontal but the second line is not
                            # Need to determine of the lines intersect on the segment in view
                            # Where does the second line intersect with the first line
                            intersection_x = (y_intercept + other_y_intercept) / other_slope

                            if A.x < B.x:
                                left_x = A.x
                                right_x = B.x
                            else:
                                left_x = B.x
                                right_x = A.x

                            # Check if other_x is greater than A.x or less than B.x
                            if left_x <= intersection_x <= right_x:
                                print("Line {} is horizontal and line {} has a slope less than 1, they should merge".format(i, j))
                    # Other line vertical check
                    elif other_slope is None:
                        # The second line is vertical but the first line is not
                        # Need to determine of the lines intersect on the segment in view
                        # Where does the second line intersect with the first line
                        intersection_y = slope * C.x + y_intercept

                        if C.y > D.y:
                            top_y = C.y
                            bottom_y = D.y
                        else:
                            top_y = D.y
                            bottom_y = C.y

                        # Check if y is greater than A.y or less than B.y
                        if top_y >= intersection_y >= bottom_y:
                            # Check if the second line slope is 50 or greater
                            # If it is, these lines should merge
                            if abs(slope) >= 50:
                                print(
                                    "Line {} has a slope greater than 50 and line {} is vertical, they should merge".format(i, j))
                    elif other_slope == 0:
                        # The first line is not horizontal but the second line is
                        # Need to determine of the lines intersect on the segment in view
                        # Where does the second line intersect with the first line
                        intersection_x = (other_y_intercept + y_intercept) / slope

                        if C.x < D.x:
                            left_x = C.x
                            right_x = D.x
                        else:
                            left_x = D.x
                            right_x = C.x

                        # Check if other_x is greater than A.x or less than B.x
                        if left_x <= intersection_x <= right_x:
                            print(
                                "Line {} has a slope less than 1 and line {} is horizontal, they should merge".format(i,j))
                    else:
                        # Both lines have a slope that is defined and not zero

                        slope_diff = abs(abs(slope) - abs(other_slope))
                        if slope_diff == 0.0:
                            # The lines are parallel
                            distance = abs(other_y_intercept - y_intercept)/math.sqrt(slope**2 + 1)

                            # Remove parallel lines that are less than 50 pixels apart
                            # if difference < 50:
                                # del line_list[j]
                                # j -= 1

                        # Else if the lines aren't parallel
                        # Check if they intersect
                        else:

                            does_intersect_on_segments = self.lineLineIntersection(A, B, C, D)

                                # If the two lines intersect on the segment visible in the frame
                                # Add the second line to a list of intersecting lines so that the middle line can be established
                                # Delete the second line from the line_list
                            if does_intersect_on_segments:
                                intersection_angle = self.calculate_intersecting_angle(slope, other_slope)
                                if intersection_angle < 10:
                                    if not found_intersecting_line:
                                        found_intersecting_line = True
                                        intersecting_lines.append(line_list[i])
                                        self.draw_line([line_list[i]], copied_frame, i)

                                    intersecting_lines.append(line_list[j])

                                    self.draw_line([line_list[j]], copied_frame, j)
                                    cv2.imshow("image", copied_frame)

                                    cv2.waitKey(200)

                                    print("Line " + str(i) + " slope: " + str(slope))
                                    print("Line " + str(j) + " slope: " + str(other_slope))
                                    print("Angle of intersection: " + str(intersection_angle))

                                    response = input("Continue")

                                    if response == 'n':
                                        sys.exit(0)
                                    # ISSUE - When lines toward the end of the list are del
                                    # del line_list[j]
                                    # j -= 1

                                    if len(line_list) <= i:
                                        i = len(line_list) -1



                j += 1

            if len(intersecting_lines) > 0:
                intersecting_lines.append(line_list[i])
                # del line_list[i]
                # i -= 1

                slopes = []
                y_intercepts = []
                for points in intersecting_lines:
                    slope = self.calculate_slope(points[0], points[1])

                    slopes.append(slope)
                    y_intercepts.append(self.calculate_y_intercept(points[0], slope))

                avg_slope = statistics.mean(slopes)
                avg_y_intercept = statistics.mean(y_intercepts)

                if self.horizontal(avg_slope):
                    x1 = 0
                    y1 = avg_y_intercept

                    x2 = 640
                    y2 = avg_slope * 640 + avg_y_intercept

                    merged_lines.append([(x1, y1), (x2, y2)])
                elif self.vertical(avg_slope):
                    x1 = (-1 * avg_y_intercept) / avg_slope
                    y1 = 0

                    x2 = (640 - avg_y_intercept) / avg_slope
                    y2 = 640

                    merged_lines.append([(x1, y1), (x2, y2)])

            i += 1

        lines_list = merged_lines


    def calculate_slope_y_intercept(self, A, B):
        if A.x == B.x:
            # This is a vertical line
            slope = None
            y_intercept = None
        else:
            slope = float(B.y - A.y) / float(B.x - A.x)
            y_intercept = A.y - slope * A.x

        return slope, y_intercept

    def calculate_intersecting_angle(self, m1, m2):
        result = abs((m1 - m2) / (1 + m1*m2))
        angle = math.atan(result)
        degrees = math.degrees(angle)
        return degrees

    def find_closest_distance(self, point1, point2, slope, y_intercept, other_slope, other_y_intercept):
        if point1.x < point2.x:
            temp = Point(point1.x, point1.y)
            end = Point(point2.x, point2.y)
        else:
            temp = Point(point2.x, point2.y)
            end = Point(point1.x, point1.y)

        closest_distance = 10**9
        while temp.x != end.x:
            first_line_y = temp.y
            second_line_y = other_slope * temp.x + other_y_intercept

            y_difference = abs(max(first_line_y, second_line_y) - min(first_line_y, second_line_y))
            closest_distance = min(closest_distance, y_difference)

            temp.x = temp.x + 1
            temp.y = slope * temp.x + y_intercept

        return closest_distance

    def calculate_slope(self, point1, point2):
        try:
            slope = float(point2[1] - point1[1]) / float(point2[0] - point1[0])
        except ZeroDivisionError:
            slope = 0.0

        return slope

    def calculate_y_intercept(self, point1, slope):
        return point1[1] - slope * point1[0]

    def horizontal(self, slope):
        return abs(slope) < 0.5

    def vertical(self, slope):
        return abs(slope) >= 0.5

    def both_horizontal(self, slope, other_slope):
        return abs(slope) < 0.5 and abs(other_slope) < 0.5

    def both_vertical(self, slope, other_slope):
        return abs(slope) >= 0.5 and abs(other_slope) >= 0.5

    def lineLineIntersection(self, A, B, C, D):
        # Line AB represented as a1x + b1y = c1
        a1 = B.y - A.y
        b1 = A.x - B.x
        c1 = a1 * (A.x) + b1 * (A.y)

        # Line CD represented as a2x + b2y = c2
        a2 = D.y - C.y
        b2 = C.x - D.x
        c2 = a2 * (C.x) + b2 * (C.y)

        determinant = a1 * b2 - a2 * b1

        if (determinant == 0):
            # The lines are parallel. This is simplified
            # by returning a pair of FLT_MAX
            return False
        else:
            x = (b2 * c1 - b1 * c2) / determinant
            y = (a1 * c2 - a2 * c1) / determinant
            return self.point_on_line(Point(x, y), A, B) and self.point_on_line(Point(x, y), C, D)

    def point_on_line(self, point, line_point_1, line_point_2):
        if line_point_1.x < line_point_2.x:
            return point.x >= line_point_1.x and point.x <= line_point_2.x
        else:
            return point.x >= line_point_2.x and point.x <= line_point_1.x

    def draw_line(self, lines_list, image, index=0):
        for line in lines_list:
            x1, y1 = line[0]
            x2, y2 = line[1]

            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # font
            font = cv2.FONT_HERSHEY_SIMPLEX

            # org
            org = (int((x1 + x2) / 2), int((y1 + y2) / 2))

            # fontScale
            fontScale = 1

            # Blue color in BGR
            color = (255, 0, 0)

            # Line thickness of 2 px
            thickness = 1

            # Using cv2.putText() method
            image = cv2.putText(image, 'Line ' + str(index), org, font,
                                 fontScale, color, thickness, cv2.LINE_AA)
            index += 1