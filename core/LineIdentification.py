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

sys.setrecursionlimit(100000)

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

        candidate_pixels = np.zeros(gray.shape)

        self.extract_side_line(frame, gray, candidate_pixels)
        self.merge_line(lines_list, frame)
        self.calibrate_camera(gray, lines_list, candidate_pixels)
        self.draw_line(lines_list, frame)
        cv2.imshow("image", frame)

        cv2.waitKey(5)
        # cv2.imwrite(frame, image)
        # modified_image = cv2.imread('modified.png')
        # cv2.imshow('Modified Image', modified_image)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # print("Done")

    def extract_side_line(self, frame, gray, candidate_pixels):


        rows, cols = gray.shape

        for c in range(cols):
            for r in range(rows):
                if candidate_pixels[r, c] == 0:
                    pixel_luminance = gray[r, c]


                    if pixel_luminance >= 200:
                        # This is a white pixel
                        # Need to search surrounding pixels to determine sideline
                        # Search should branch outward from central pixel until a non-white pixel is found
                        # Search left

                        right_pixel_count = 0
                        for right in range(c + 1, cols):
                            if gray[r, right] >= 200:
                                right_pixel_count += 1
                            else:
                                break

                        left_pixel_count = 0
                        for left in range(c-1, -1, -1):
                            other_luminance = gray[r, left]
                            if gray[r, left] >= 200:
                                left_pixel_count += 1
                            else:
                                break

                        top_pixel_count = 0
                        for top in range(r-1, -1, -1):
                            if gray[top, c] >= 200:
                                top_pixel_count += 1
                            else:
                                break

                        bottom_pixel_count = 0
                        for bottom in range(r + 1, rows):
                            if gray[bottom, c] >= 200:
                                bottom_pixel_count += 1
                            else:
                                break

                        if max(left_pixel_count, right_pixel_count) > 50\
                                and max(top_pixel_count, bottom_pixel_count) > 10:
                            # Assume this is a sideline pixel
                            # Starting from this pixel, radiate out until a non white pixel is found in a circular fashion
                            self.color_pixels(gray, frame, candidate_pixels, r, c)

        for c in range(cols):
            for r in range(rows):
                if candidate_pixels[r, c] == 1:

                    right_pixel_count = 0
                    for right in range(c + 1, cols):
                        if gray[r, right] >= 200:
                            right_pixel_count += 1
                        else:
                            break

                    left_pixel_count = 0
                    for left in range(c-1, -1, -1):
                        other_luminance = gray[r, left]
                        if gray[r, left] >= 200:
                            left_pixel_count += 1
                        else:
                            break

                    top_pixel_count = 0
                    for top in range(r - 1, -1, -1):
                        if gray[top, c] >= 200:
                            top_pixel_count += 1
                        else:
                            break

                    bottom_pixel_count = 0
                    for bottom in range(r + 1, rows):
                        if gray[bottom, c] >= 200:
                            bottom_pixel_count += 1
                        else:
                            break

                    if right_pixel_count < 5 and left_pixel_count < 5:
                        candidate_pixels[r, c] = 0

        for c in range(cols):
            for r in range(rows):
                if candidate_pixels[r, c] == 1:
                    x = c
                    y = r
                    cv2.circle(frame, (x, y), 0, (0, 255, 0), 1)


    def color_pixels(self, gray, copied_img, candidate_pixels, r, c):
        rows, cols = gray.shape

        if r < 0 or r >= rows or c < 0 or c >= cols:
            return 0

        if gray[r, c] < 200:
            return 0

        if candidate_pixels[r, c] == 1:
            return 0

        candidate_pixels[r, c] = 1



        self.color_pixels(gray, copied_img, candidate_pixels, r + 1, c)
        self.color_pixels(gray, copied_img, candidate_pixels, r + 1, c + 1)
        self.color_pixels(gray, copied_img, candidate_pixels, r, c + 1)
        self.color_pixels(gray, copied_img, candidate_pixels, r - 1, c + 1)
        self.color_pixels(gray, copied_img, candidate_pixels, r - 1, c)
        self.color_pixels(gray, copied_img, candidate_pixels, r - 1, c - 1)
        self.color_pixels(gray, copied_img, candidate_pixels, r, c - 1)
        self.color_pixels(gray, copied_img, candidate_pixels, r + 1, c - 1)

    def calibrate_camera(self, frame, line_list, sideline_pixels):
        rows, cols = frame.shape

        midpoint_x = int(cols / 2)
        midpoint_y = int(rows / 2)


        # Iterate up/down to find the point at which this intersects with a horizontal line
        top_sideline = (None, None)
        bottom_sideline = (None, None)

        for line in line_list:
            A = Point(line[0][0], line[0][1] * -1)
            B = Point(line[1][0], line[1][1] * -1)

            slope, y_intercept = self.calculate_slope_y_intercept(A, B)

            if self.horizontal(slope):
                y = slope * midpoint_x + y_intercept

                for r in reversed(range(midpoint_y)):
                    if r * -1 == y:
                        if top_sideline is (None, None):
                            top_sideline = (line, r)
                        else:
                            if top_sideline[1] < r:
                                top_sideline = (line, r)

                for r in range(midpoint_y, rows):
                    if r * -1 == y:
                        if bottom_sideline is (None, None):
                            bottom_sideline = (line, r)
                        else:
                            if bottom_sideline[1] > r:
                                bottom_sideline = (line, r)

        self.extend_lines_to_edge(top_sideline, cols)
        self.extend_lines_to_edge(bottom_sideline, cols)

        copied_gray = frame.copy()
        self.draw_line([top_sideline[0], bottom_sideline[0]], copied_gray)
        cv2.imshow("line", copied_gray)
        cv2.waitKey(0)
        print("lines")

    def extend_lines_to_edge(self, line, cols):
        temp_A = Point(line[0][0][0], line[0][0][1] * -1)
        temp_B = Point(line[0][1][0], line[0][1][1] * -1)

        if temp_A.x < temp_B.x:
            if temp_A != 0:
                line[0][0] = (0, line[0][0][1])
            if temp_B != cols:
                line[0][1] = (cols, line[0][1][1])
        else:
            if temp_B.x != 0:
                line[0][1] = (0, line[0][1][1])
            if temp_A.x != cols:
                line[0][0] = (cols, line[0][0][1])

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
                            # Check if the distance between them is less than 5
                            distance = abs(A.x - C.x)

                            if distance < 5:
                                # print ("Line {} and {} are both vertical and should merge".format(i, j))
                                intersecting_lines.append(line_list[j])

                                if i > j:
                                    i -= 1

                                del (line_list[j])
                                j -= 1
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
                                if abs(other_slope) >= 5:
                                    # print("Line {} is vertical and line {} has a slope greater than 50, they should merge".format(i, j))
                                    intersecting_lines.append(line_list[j])

                                    if i > j:
                                        i -= 1

                                    del (line_list[j])
                                    j -= 1
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
                            if abs(slope) >= 5:
                                # print(
                                #     "Line {} has a slope greater than 50 and line {} is vertical, they should merge".format(i, j))
                                intersecting_lines.append(line_list[j])

                                if i > j:
                                    i -= 1

                                del (line_list[j])
                                j -= 1
                    # Horizontal line check
                    elif slope == 0:
                        if other_slope == 0:
                            # Both these lines are horizontal
                            distance = abs(A.y - C.y)

                            if distance < 5:
                                # print ("Line {} and {} are both horizontal and should merge".format(i, j))
                                intersecting_lines.append(line_list[j])

                                if i > j:
                                    i -= 1

                                del (line_list[j])
                                j -= 1
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
                                # print("Line {} is horizontal and line {} has a slope less than 1, they should merge".format(i, j))
                                intersecting_lines.append(line_list[j])

                                if i > j:
                                    i -= 1

                                del (line_list[j])
                                j -= 1

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
                            # print(
                            #     "Line {} has a slope less than 1 and line {} is horizontal, they should merge".format(i,j))
                            intersecting_lines.append(line_list[j])

                            if i > j:
                                i -= 1

                            del (line_list[j])
                            j -= 1
                    else:
                        # Both lines have a slope that is defined and not zero

                        slope_diff = abs(abs(slope) - abs(other_slope))

                        if slope_diff == 0.0:
                            # The lines are parallel
                            distance = abs(other_y_intercept - y_intercept)/math.sqrt(slope**2 + 1)

                            # Remove parallel lines that are less than 50 pixels apart
                            if distance < 50:
                                intersecting_lines.append(line_list[j])

                                if i > j:
                                    i -= 1

                                del (line_list[j])
                                j -= 1
                                # del line_list[j]
                                # j -= 1

                        # Else if the lines aren't parallel
                        # Check if the slop difference is less than 0.5
                        # Check if they intersect
                        elif slope_diff < 0.5:
                            does_intersect_on_segments = self.lineLineIntersection(A, B, C, D)

                            if does_intersect_on_segments:
                                # If the two lines intersect on the segment visible in the frame
                                # Add the second line to a list of intersecting lines so that the middle line can be established
                                # Delete the second line from the line_list
                                intersection_angle = self.calculate_intersecting_angle(slope, other_slope)
                                if intersection_angle < 10:

                                    intersecting_lines.append(line_list[j])

                                    if i > j:
                                        i -= 1

                                    del (line_list[j])
                                    j -= 1

                                    # print("Line " + str(i) + " slope: " + str(slope))
                                    # print("Line " + str(j) + " slope: " + str(other_slope))
                                    # print("Angle of intersection: " + str(intersection_angle))
                                    # ISSUE - When lines toward the end of the list are del
                                    # del line_list[j]
                                    # j -= 1

                            else:
                                # The lines do not intersect the segment
                                # There are cases where lines represent the same/similar edges but are not parallel and
                                # do not intersect on the segment
                                # For these lines, at a given point (abitrarily the midpoint of the first line),
                                # the distance between the two lines is less than 5
                                midpoint_x = (A.x + B.x) / 2
                                midpoint_y = (A.y + B.y) / 2

                                other_y = other_slope * midpoint_x + other_y_intercept

                                distance = math.sqrt((abs(other_y) - abs(midpoint_y)) ** 2)

                                temp_lines = [line_list[i], line_list[j]]


                                if distance < 50:

                                    intersecting_lines.append(line_list[j])

                                    if i > j:
                                        i -= 1

                                    del(line_list[j])
                                    j -= 1

                                    # print("Line " + str(i) + " slope: " + str(slope))
                                    # print("Line " + str(j) + " slope: " + str(other_slope))
                                    # print("Distance: " + str(distance))
                        elif abs(slope) > 15 and abs(other_slope) > 15:

                            # print("{} - {}".format(i, j))

                            # The lines do not intersect the segment
                            # There are cases where lines represent the same/similar edges but are not parallel and
                            # do not intersect on the segment
                            # For these lines, at a given point (abitrarily the midpoint of the first line),
                            # the distance between the two lines is less than 5
                            midpoint_x = (A.x + B.x) / 2
                            midpoint_y = (A.y + B.y) / 2

                            other_y = other_slope * midpoint_x + other_y_intercept

                            distance = math.sqrt((abs(other_y) - abs(midpoint_y)) ** 2)

                            temp_lines = [line_list[i], line_list[j]]

                            if distance < 100:

                                intersecting_lines.append(line_list[j])

                                if i > j:
                                    i -= 1

                                del (line_list[j])
                                j -= 1

                                # print("Line " + str(i) + " slope: " + str(slope))
                                # print("Line " + str(j) + " slope: " + str(other_slope))
                                # print("Distance: " + str(distance))

                j += 1

            if len(intersecting_lines) > 0:
                intersecting_lines.append(line_list[i])
                slopes = []
                y_intercepts = []

                is_vertical = False
                for points in intersecting_lines:
                    temp_A = Point(points[0][0], points[0][1])
                    temp_B = Point(points[1][0], points[1][1])
                    slope, y_intercept = self.calculate_slope_y_intercept(temp_A, temp_B)

                    if slope is None and not is_vertical:
                        is_vertical = True

                    slopes.append(slope)
                    y_intercepts.append(y_intercept)

                if is_vertical:
                    x1  = 0
                else:
                    avg_slope = statistics.mean(slopes)
                    avg_y_intercept = statistics.mean(y_intercepts)

                    if self.horizontal(avg_slope):



                        x1 = 0
                        y1 = int(avg_y_intercept)

                        x2 = 640
                        y2 = int(avg_slope * 640 + avg_y_intercept)
                    else:
                        x1 = int((-1 * avg_y_intercept) / avg_slope)
                        y1 = 0

                        x2 = int((640 - avg_y_intercept) / avg_slope)
                        y2 = 640

                    new_line = [(x1, y1), (x2, y2)]
                    merged_lines.append(new_line)

                    line_list[i] = new_line
                    i = 0
            else:
                merged_lines.append(line_list[i])

                i += 1




    def calculate_slope_y_intercept(self, A, B):
        if A.x == B.x:
            # This is a vertical line
            slope = None
            y_intercept = None
        else:
            if A.x < B.x:
                slope = float(B.y - A.y) / float(B.x - A.x)
                y_intercept = A.y - slope * A.x
            else:
                slope = float(A.y - B.y) / float(A.x - B.x)
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