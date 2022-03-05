""" grab_stills.py

Utility script to grab still images from videos, labeling them with timestamps in microseconds.
Intended for grabbing stills from catalysis images for labeling (ie annotating).

Code snippets borrowed from https://www.geeksforgeeks.org/extract-images-from-video-in-python/

last updated 3/4/22 by Aristana Scourtas
"""

import cv2
import math
import os

video_name = "IrO2_DSC_0106"
dest_folder = "./still_data"
source_video_path = "/Users/aristanascourtas/Documents/Work/bubble misc/Test Videos-2-28-2022/DSC_0106.MOV"

cam = cv2.VideoCapture(source_video_path)
# frame
currentframe = 0

# get video FPS (e.g. ~60 fps)
fps = cam.get(cv2.CAP_PROP_FPS)

# the seconds between frame saves
save_rate = 30

# calculate the number of n frames in each save_rate interval
n_frame_to_grab = save_rate * math.ceil(fps)


try:
    # create a local folder to save the data
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
except OSError:
    print('Error: Creating directory of data')

while True:
    # reading from frame
    ret, frame = cam.read()
    # timestamp, in microseconds, rounded to avoid periods in filename
    timestamp = round(cam.get(cv2.CAP_PROP_POS_MSEC) * 1000)

    # only grab every nth frame
    if currentframe % n_frame_to_grab == 0.0:
        # if video is still left continue writing images
        if ret:
            filename = f"{video_name}_{str(timestamp)}.jpg"
            dest_path = os.path.join(dest_folder, filename)
            print('Writing ' + dest_path)

            # write the extracted images to disk
            cv2.imwrite(dest_path, frame)
        else:
            break
    currentframe += 1

# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()
