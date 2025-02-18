from core.LineIntersections import LineIdentification

import cv2

if __name__ == '__main__':
    li = LineIdentification()
    img = cv2.imread('toney-all-22.jpg')

    li.edge_detection(img)

    cv2.imshow('Lines', img)

    cv2.waitKey(0)

