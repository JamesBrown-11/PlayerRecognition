
import cv2
import os
import sys
from core.LineIdentification import LineIdentification

base_dir = os.path.join(os.getcwd(), "data", "field")
train_labels = []
train_images = []
output_dataset = []


def init():
    global base_dir, train_labels, train_images, base_dir

    if not os.path.exists(base_dir):
        print("Base directory " + base_dir + " doesn't exist")
        sys.exit()

    if not os.path.exists(os.path.join(base_dir, "train", "classes")):
        os.makedirs(os.path.join(base_dir, "train", "classes"))

    train_labels = os.listdir(os.path.join(base_dir, "train", "labels"))
    train_images = os.listdir(os.path.join(base_dir, "train", "images"))

    for i in range(len(train_labels)):
        annotate(train_labels[i], train_images[i])




def annotate(file, image):
    global output_dataset

    img = cv2.imread(os.path.join(base_dir, "train", "images", image))

    line_identifier = LineIdentification()
    line_identifier.edge_detection(img, threshold=150, minLineLength=50, maxLineGap=100)

    with open(os.path.join(base_dir, "train", "labels", file), "r") as f:
        for line in f:
            label, x_center, y_center, w, h = line.split(" ")
            normalized_x_center = float(x_center) * 640
            normalized_y_center = float(y_center) * 640
            normalized_w = float(w) * 640
            normalized_h = float(h) * 640

            top_left = (int(normalized_x_center - (normalized_w/2)), int(normalized_y_center - (normalized_h/2)))
            bottom_right = (int(normalized_x_center + (normalized_w/2)), int(normalized_y_center + (normalized_h/2)))

            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

            # cv2.imshow("image", img)

            # cv2.waitKey(200)

            # field_location = record_field_location()

            # output_dataset.append(create_sample(image, line, field_location))
        choice = input("Enter 1 to stop or 0 to continue")
        if choice == "1":
            write_dataset()
            sys.exit()
        cv2.destroyAllWindows()

def record_field_location():
    print("What is the x coordinate of this bounding box?")
    x = input()

    print("What is the y coordinate of this bounding box?")
    y = input()

    return int(x), int(y)


def create_sample(image, line, field_location):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    box_coords = line.split(" ")

    features = ""

    for val in grayscale:
        features += val + " "

    for val in box_coords:
        features += val + " "

    features += field_location[0] + " " + field_location[1]

    return features


def write_dataset():
    global output_dataset

    with open(os.path.join(base_dir, "train", "classes", "dataset.txt"), "w") as f:
        for line in output_dataset:
            f.write(line + "\n")

if __name__ == "__main__":
    init()