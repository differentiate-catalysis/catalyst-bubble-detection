""" grab_stills.py

Utility script to grab still images from videos, labeling them with timestamps in microseconds.
Intended for grabbing stills from catalysis images for labeling (ie annotating).

Code snippets borrowed from https://www.geeksforgeeks.org/extract-images-from-video-in-python/

last updated 3/4/22 by Aristana Scourtas
"""

from argparse import ArgumentParser
import cv2
import math
import os


def write_still_images(video_name, source_video_path, dest_folder, save_rate):
    """Grab still images from source video and write them to disk

    video_name (str): The shortname for the video to process, which will be written into the still
    image filenames

    source_video_path (str): The full filepath to the source video to process

    dest_folder (str): The path of the folder to which the still images will be written

    save_rate (int): The number of seconds between each frame to save
    """

    cam = cv2.VideoCapture(source_video_path)
    # frame counter
    currentframe = 0

    # get video FPS (e.g. ~60 fps)
    fps = cam.get(cv2.CAP_PROP_FPS)

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


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Utility script to grab still images from videos, labeling them with timestamps "
                    "in microseconds. Intended for grabbing stills from catalysis images for labeling "
                    "(ie annotating).")
    parser.add_argument("--short_name", dest="video_name", type=str, required=True,
                        help="The shortname for the video to process, which will be written into the still image "
                             "filenames. Ex: IrO2_DSC_0106")
    parser.add_argument("--source_path", dest="source_video_path", type=str, required=True,
                        help="The full filepath to the source video to process.")
    parser.add_argument("--dest", dest="dest_folder", type=str, default="./still_data",
                        help="The path of the folder to which the still images will be written. Default is "
                             "'./still_data'")
    parser.add_argument("--save_rate", dest="save_rate", type=int, default=30,
                        help="The number of seconds between each frame to save. Default is 30. "
                             "Ex: 30 would mean saving 1 frame every 30s")

    args = parser.parse_args()

    write_still_images(args.video_name, args.source_video_path, args.dest_folder, args.save_rate)
