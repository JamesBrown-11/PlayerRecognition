from core.LineIntersections import LineIdentification
import os
import cv2


if __name__ == '__main__':
    extractor = LineIdentification()

    directory = '../data/dataset1'
    extraction_directory = '../data/dataset1_edges'

    # Create the directory
    try:
        os.mkdir(extraction_directory)
        print(f"Directory '{extraction_directory}' created successfully.")
    except FileExistsError:
        print(f"Directory '{extraction_directory}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{extraction_directory}'.")
        os.exit(-1)
    except Exception as e:
        print(f"An error occurred: {e}")
        os.exit(-1)

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)

        if os.path.isfile(f):
            img = cv2.imread(f)

            edges = extractor.edge_detection(img)

            file_components = filename.split('.')

            filename_edges = file_components[0] + '_edges.' + file_components[1]

            output_path = os.path.join(extraction_directory, filename_edges)

            cv2.imwrite(output_path, edges)

