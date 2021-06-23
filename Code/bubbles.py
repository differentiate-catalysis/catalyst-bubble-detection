"""
Bubble-tracking Mask R-CNN code

Written by @srufer 09/2020

Adapted from:
bubble.py
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 bubble.py train --dataset=/path/to/bubble/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 bubble.py train --dataset=/path/to/bubble/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 bubble.py train --dataset=/path/to/bubble/dataset --weights=imagenet

    # Apply color splash to an image
    python3 bubble.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 bubble.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

import time

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize


# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



############################################################
# Global Variables for tracking bubble instances from one frame to the next
############################################################

maxBubblesPerFrame = 150                    # length of bubble tracking array
cols = 7                                    # width of bubble tracking array (number of datapoints for each bubble)
curr = np.zeros((maxBubblesPerFrame, cols)) # current frame bubble tracking array
prev = np.zeros((maxBubblesPerFrame, cols)) # previous frame bubble tracking array

lifetime_numBubbles = 1                     # counter for how many unique bubbles have been identified

precision_ID = 10                           # number of pixels the center of a bubble can drift from one frame to the next without triggering a new instance
precision_coalesce = 10                     # If a bubble dissapears, all bubbles within this number of pixels are assumed to have participated in coalesence
limit_nucleation = 23                       # If a newly identified bubble has a diameter greater than this number (in pixels) its growth is ignored (must not be a
                                            # newly nucleated bubble, rather a large bubble that was lost in the previous frame and then re-recognized)

counter = 0

############################################################
#  Configurations
############################################################


class BubbleConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "bubble"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + bubble

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.8


############################################################
#  Dataset
############################################################

class BubbleDataset(utils.Dataset):

    def load_bubble(self, dataset_dir, subset):
        """Load a subset of the bubble dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("bubble", 1, "bubble")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "bubble",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a bubble dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "bubble":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "bubble":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = BubbleDataset()
    dataset_train.load_bubble(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BubbleDataset()
    dataset_val.load_bubble(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, [0,150,170], gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    
    return splash


############################################################
#  Bubble Tracking Algorithms
############################################################

def bounding_circle(image, boxes):
    
    global maxBubblesPerFrame
    global rows
    global curr
    global prev
    global lifetime_numBubbles
    global precision_ID
    global precision_coalesce
    global counter
    
    import cv2
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")

    for i in range(N):
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]   # corners of bounding box. The accuracy of this algorithm could be greatly improved by  interpreting the masks directly as opposed to using the bounding box

        D = int((x2+y2-x1-y1)/2)    # width of bounding box == diameter of bubble
        center_x = int((x2+x1)/2)
        center_y = int((y2+y1)/2)
        match_plusminus = 0

        image = np.array(image)
        if D < 200 and i < maxBubblesPerFrame:    #bubble found!!
            #Image editing
            image = image.astype(np.uint8)
        
        
# instance growth tracking: calculating the size +- for each bubble
            # bubble tracking array
            # | 0  | 1         | 2         | 3         | 4    | 5          | 6       |
            # | ID | center X  | center Y  | Diameter  | New? | Coalesced? | Size +- |
            curr[i, 1] = center_x
            curr[i, 2] = center_y
            curr[i, 3] = D

            # go through the previous bubble tracking array and see if the current instance existed in the previous frame
            foundmatch = 0
            match_row = 0
            for j in range(maxBubblesPerFrame):
                if (np.abs(prev[j, 1]-center_x) + np.abs(prev[j, 2] - center_y)) < precision_ID:    # match based on center position
                    foundmatch = 1
                    curr[i, 0] = prev[j, 0] #assign id's
                    match_row = j
                    match_plusminus = prev[j, 6]
                    break

            # calculate Size +-
            if not foundmatch: #newly identified bubble!
                curr[i, 4] = 1 #set new flag
                curr[i, 0] = lifetime_numBubbles
                lifetime_numBubbles = lifetime_numBubbles + 1
                if (D < limit_nucleation): # check if "new" bubble is small enough to actually be a new bubble as opposed to a lost + refound bubble.
                    curr[i, 6] = 3.14/6*D*D*D # volume of a sphere
                else:
                    curr[i, 6] = 0 #change its +- to zero
                    curr[i, 4] = 0 #remove its fresh tag

            else: #not a new bubble
                curr[i, 4] = 0
                curr[i, 6] = 3.14/6*(D*D*D - prev[match_row, 3]*prev[match_row, 3]*prev[match_row, 3])      # incremental growth from the previous frame
                    
        else:
            print("ignored (too big or too many)!")

# Coalesence invalidation: if a bubble participates in a coalesence event, it should not contribute to the Size +-
    for p in range(maxBubblesPerFrame):
        # check if each bubble from prev still exists in curr
        coalesce = 1
        for c in range(maxBubblesPerFrame):
            if(prev[p, 0] == curr[c, 0]):   #bubble still exists
                coalesce = 0
                break

        if(coalesce):
            for c2 in range(maxBubblesPerFrame):
                dx = np.abs(prev[p, 1]-curr[c2, 1])
                dy = np.abs(prev[p, 2]-curr[c2, 2])
                dd = np.sqrt(dx*dx + dy*dy)             # distance between bubble centers
                
                avg_D = int((prev[p, 3] + curr[c2, 3])/2)
                
                if (dd - avg_D < precision_coalesce):   # If a bubble is within precision_coalesence of the dissapeared bubble, it should not contribute to Size +-
                    curr[c2, 5] = 1   # coalesence flag
#                   curr[c2, 6] = 0  # zeroth order
                    curr[c2, 6] = match_plusminus  # first order (take the plus/minus from the last frame)


# summing all the Size +- and assigning colors
    sum = 0
    for s in range(maxBubblesPerFrame):
        sum = sum + curr[s, 6]

        color = (255, 255, 255)
        if (curr[s, 4]):  #New bubble!
            color = (224, 230, 73)
        elif (curr[s, 5]):  #coalesced bubble
            color = (96, 66, 245)
        elif (curr[s, 6] > 0): #growing bubble
            color = (50, 168, 82)
        elif(curr[s, 6] == 0): # size unchanged
            color = (179, 247, 255)
        else:   #shrinking bubble
            color = (150, 0, 0)

        # draw circle and write ID on frame
        cv2.putText(image, str(int(curr[s, 0])), (int(curr[s, 1])-9, int(curr[s, 2])+7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.circle(image, (int(curr[s, 1]), int(curr[s, 2])), int(curr[s, 3]/2), color, 3)

    with open("res.csv", 'a') as f: # log the total size+- for the counter-th frame
        f.write(str(counter) + ", " + str(sum))
        f.write("\n")


# Resetting
    np.savetxt("./ID_table/" + str(counter) + ".csv", curr, delimiter=",")  # save all data to csv
    prev = curr
    curr = np.zeros((maxBubblesPerFrame, cols))
    counter = counter + 1
# end Resetting

    return image


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        start = time.time()
        r = model.detect([image], verbose=1)[0]
        end = time.time()
        print("Detect time: ")
        print(end-start)
        print("\n")
        #Bounding Circle
        splash = bounding_circle(image, r['rois'])
        # Color splash
        #splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                
                # Detect objects
                start = time.time()
                r = model.detect([image], verbose=0)[0]
                end = time.time()
                print("Detect time: ")
                print(end-start)
                print("\n")
                
                # Bounding circle
                splash = bounding_circle(image, r['rois'])
                
                # Color Splash
#                splash = color_splash(image, r['masks'])


                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
    
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect bubbles.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/bubble/dataset/",
                        help='Directory of the bubble dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BubbleConfig()
    else:
        class InferenceConfig(BubbleConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

#Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))


###bounding box style
#image = skimage.io.imread(args.image)
#
## Run detection
#results = model.detect([image], verbose=1)
#
## Visualize results
#r = results[0]
#visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
#                            ['background','bubble'], r['scores'], show_mask=False)





