""" grab_stills.py

Utility script to grab still images from videos, labeling them with timestamps.
Intended for grabbing stills from catalysis images for labeling (ie annotating) for training.

Code snippets borrowed from https://www.geeksforgeeks.org/extract-images-from-video-in-python/

last updated 3/4/22 by Aristana Scourtas
"""

import cv2
import math
import os

SAVING_FPS = 0.1  # this would be one still every 10s

cam = cv2.VideoCapture("/Users/aristanascourtas/Documents/Work/bubble misc/Test Videos-2-28-2022/DSC_0106.MOV")

try:

    # creating a folder named data
    if not os.path.exists('still_data'):
        os.makedirs('still_data')

# if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0


# get video FPS (~60 fps)
fps = cam.get(cv2.CAP_PROP_FPS)

# grab a frame every nth second
n_frame_to_grab = 30 * math.ceil(fps)


while(True):
    # reading from frame
    ret, frame = cam.read()

    # only grab every nth frame
    if currentframe % n_frame_to_grab == 0.0:
        if ret:

            # if video is still left continue creating images
            name = './still_data/frame' + str(currentframe) + '.jpg'
            print('Creating...' + name)

            # writing the extracted images
            cv2.imwrite(name, frame)
        else:
            break
    # increasing counter so that it will
    # show how many frames are created
    currentframe += 1

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()


