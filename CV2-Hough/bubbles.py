import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

def hough_circles(img, param1, param2):
    src = cv2.cuda_GpuMat()
    src.upload(img)
    # dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius, maxCircles
    detector = cv2.cuda.createHoughCirclesDetector(1, 50, param1, param2, 10, 200, 4096)
    circles = detector.detect(src)
    circles_cpu = circles.download()
    return circles_cpu

if __name__ == '__main__':
    img = cv2.imread('aluminum_reflective10.png')
    blurred = cv2.medianBlur(img, 3)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    for i in range(10, 201, 5):
        for j in range(10, 201, 5):
            circles_cpu = hough_circles(gray, i, j)
            if circles_cpu is not None:
                print('%d, %d, %d' % (i, j, len(circles_cpu[0])))
            else:
                print('%d, %d, 0' % (i, j))
