import argparse
import json
import os

import cv2 as cv
import numpy as np

from utils import tqdm


def label_to_bubble_json(pred_file, json_file: str=None, label_file: str=None):
    """UNUSED: Uses HoughCircles to convert a label file (i.e. a semantic segmentation output) to a list of bubbles.
    Args:
        pred_file (str): Image file with segmentation output
        json_file (str, optional): JSON file to store output. Defaults to None.
        label_file (str, optional): Image file to save circled detected bubbles to. Defaults to None.
    Returns:
        _type_: _description_
    """
    img = cv.imread(pred_file,0)
    img = cv.GaussianBlur(img, (3,3), cv.BORDER_DEFAULT)
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,20,
                                param1=45,param2=20,minRadius=10,maxRadius=150)
    circles = np.uint16(np.around(circles))

    outdir = {}
    outdir['width'] = img.shape[0]
    outdir['height'] = img.shape[1]
    outdir['filename'] = os.path.basename(pred_file)
    outdir['bubbles'] = []
    for circle in circles[0]:
        bubble_data = {
            "center": [circle[0].item(), circle[1].item()],
            "radius": circle[2].item()
        }
        outdir['bubbles'].append(bubble_data)
    if label_file:
        cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
        cv.imwrite(label_file, cimg)
    if json_file:
        with open(json_file, 'w') as outfp:
            json.dump(outdir, outfp, indent=2)
    return outdir


def prediction_to_bubble_dir(pred_dir, json_dir = None, label_dir = None):
    if json_dir and not os.path.isdir(json_dir):
        os.makedirs(json_dir)
    if label_dir and not os.path.isdir(label_dir):
        os.makedirs(label_dir)
    files = [f for f in os.listdir(pred_dir) if os.path.isfile(os.path.join(pred_dir, f))]
    for file in tqdm(files):
        json_file = os.path.join(json_dir, os.path.splitext(file)[0] + '.json') if json_dir else None
        label_file = os.path.join(label_dir, file) if label_dir else None
        label_to_bubble_json(os.path.join(pred_dir, file), json_file, label_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to directory with predictions')
    parser.add_argument('--json_dir', type=str, help='Path to directory to store json output')
    parser.add_argument('--label_dir', type=str, help='Path to directory to store labeled images')

    args = parser.parse_args()

    prediction_to_bubble_dir(args.pred_dir, args.json_dir, args.label_dir)
