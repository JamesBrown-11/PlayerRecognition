from core.LineIntersections import LineExtractor
import os
import cv2
import shutil


def create_dir (dir_path) :
    # Create the directory
    try:
        os.mkdir(dir_path)
        print(f"Directory '{dir_path}' created successfully.")
    except FileExistsError:
        print(f"Directory '{dir_path}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{dir_path}'.")
        os.exit(-1)
    except Exception as e:
        print(f"An error occurred: {e}")
        os.exit(-1)


if __name__ == '__main__':
    extractor = LineExtractor()

    directory = '../data/dataset1'
    endzone_path = '../data/dataset1_edges_endzone'
    sideline_path = '../data/dataset1_edges_sideline'

    create_dir(endzone_path)
    create_dir(sideline_path)

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        if os.path.isfile(f):
            # Read image
            img = cv2.imread(f)

            # Extract the edges only from image
            edges = extractor.edge_detection(img)

            # img = extractor.draw_lines(img, edges)

            # Determine if the image is from the end zone or sideline
            # cv2.imshow("edges", edges)
            # cv2.waitKey(5)
            orientation = extractor.calculate_orientation(edges)
            # cv2.imshow("img", img)

            # cv2.waitKey(5)

            if orientation == 'endzone':
                # This image is a endzone positioned image
                output_path = os.path.join(endzone_path, filename)
                cv2.imwrite(output_path, img)

            elif orientation == 'sideline':
                # This image is a sideline positioned image
                output_path = os.path.join(sideline_path, filename)
                cv2.imwrite(output_path, img)
            else:
                print(filename + " ignored")

