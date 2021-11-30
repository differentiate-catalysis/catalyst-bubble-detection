import math
import sys
from typing import List, Tuple
import os
from utils import get_iou_types

import torch
import torch.utils.data
import torchvision
from ray import tune
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
import skimage.io
import cv2
import numpy as np

from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

from models import model_mappings
from utils import Dataset, collate_fn

def evaluate(model: Module, valid_loader: DataLoader, amp: bool, gpu: int, save_dir: str = None, test: bool = False, apply: bool = True, metrics: bool = True) -> Tuple[float, float, float]:
    if gpu != -1:
        device = torch.device('cuda', gpu)
    else:
        device = torch.device('cpu')

    with torch.no_grad():
        # Set model to evaluate mode (disable batchnorm and dropout, output boxes)
        model.eval()
        total_loss = 0
        num_samples = valid_loader.batch_size * len(valid_loader)
        boxes = []
        target_boxes = []
        scores = []
        _, has_target = next(iter((valid_loader)))
        if has_target[0] is not None:
            coco = get_coco_api_from_dataset(valid_loader.dataset)
            iou_types = get_iou_types(model)
            coco_evaluator = CocoEvaluator(coco, iou_types)
        for j, (images, targets) in enumerate(tqdm(valid_loader)):
            images = list(image.to(device) for image in images)
            has_target = targets[0] is not None
            if has_target:
                targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(images)
                if has_target:
                    res = {}
                    for target, output in zip(targets, outputs):
                        # NMS the boxes prior to evaluation
                        indices = torchvision.ops.nms(output['boxes'], output['scores'], 0.3)
                        output = {k: v[indices].cpu() for k, v in output.items()}
                        res[target['image_id'].item()] = output
                        boxes.append(output['boxes'].to(device))
                        scores.append(output['scores'].to(device))
                        target_boxes.append(target['boxes'].to(device))

                        if test:
                            this_image = os.path.basename(os.listdir(valid_loader.dataset.patch_root)[j])
                            if 'boxes' in output:
                                box_cpu = output['boxes'].cpu()
                                np.save(os.path.join(save_dir, 'boxes', this_image), box_cpu)
                            if 'masks' in output:
                                mask_cpu = output['masks'].cpu()
                                np.save(os.path.join(save_dir, 'masks', this_image), mask_cpu)
                            #if scores:
                                #np.save(os.path.join(save_dir, 'scores', this_image), scores)
                            labeled = label_on_image(os.path.join(valid_loader.dataset.patch_root, this_image), output['boxes'])
                            label_image = to_pil_image(labeled)
                            label_image.save(os.path.join(save_dir, 'labeled_images', this_image))
                    coco_evaluator.update(res)

                    # Must set model to train mode to get loss
                    model.train()
                    loss_dict = model(images, targets)
                    loss = sum(loss for loss in loss_dict.values()).item()
                    model.eval()
                    if not math.isfinite(loss) and not test:
                        print('Loss is %s, stopping training' % loss)
                        if tune.is_session_enabled():
                            tune.report(loss=10000, iou=0, map=0)
                        return 10000, 0, 0
                    total_loss += loss * valid_loader.batch_size
                elif test:
                    for output in outputs:
                        this_image = os.path.basename(os.listdir(valid_loader.dataset.patch_root)[j])
                        if 'boxes' in output:
                            indices = torchvision.ops.nms(output['boxes'], output['scores'], 0.3)
                            nms_boxes = output['boxes'][indices]
                            box_cpu = output['boxes'].cpu()
                            np.save(os.path.join(save_dir, 'boxes', this_image), box_cpu)
                            labeled = label_on_image(os.path.join(valid_loader.dataset.patch_root, this_image), nms_boxes)
                            label_image = to_pil_image(labeled)
                            label_image.save(os.path.join(save_dir, 'labeled_images', this_image))
                        if 'masks' in output:
                            mask_cpu = output['masks'].cpu()
                            np.save(os.path.join(save_dir, 'masks', this_image), mask_cpu)
                        if scores:
                            np.save(os.path.join(save_dir, 'scores', this_image), scores)
        if has_target:
            coco_evaluator.synchronize_between_processes()
            coco_evaluator.accumulate()
            stats = coco_evaluator.summarize()
        if not test:
            mAP, iou = get_metrics(boxes, scores, target_boxes, device, 0.5)
            mAP = stats[0][0]
            print('--- evaluation result ---')
            print('loss: %.5f, mAP %.5f, IoU %.5f' % (total_loss / num_samples, mAP, iou))

            return total_loss / num_samples, iou, mAP
        else:
            return total_loss / num_samples

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    scores, indices = torch.sort(scores, dim=0, descending=True)
    boxes = boxes[indices, :]
    ious = torchvision.ops.box_iou(boxes, boxes)
    for i, box in enumerate(ious):
        bad_boxes = box >= iou_threshold
        bad_boxes[i] = False
        ious[bad_boxes] = torch.zeros_like(box)
    ious = torch.sum(ious, dim=-1).flatten()
    boxes = boxes[ious > 0, :]
    scores = scores[ious > 0]
    return boxes, scores


def get_metrics(boxes: List[Tensor], scores: List[Tensor], targets: List[Tensor], device: torch.device, iou_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    # Map each box index to its image
    label_images = []
    for i in range(len(targets)):
        label_images.extend([i] * targets[i].shape[0])
    label_images = torch.tensor(label_images, device=device, dtype=torch.long)

    detection_images = []
    for i in range(len(boxes)):
        detection_images.extend([i] * boxes[i].shape[0])
    detection_images = torch.tensor(detection_images, device=device, dtype=torch.long)

    # Replacement for flatten, cannot flatten since non-rectangular
    targets = torch.cat(targets, dim=0)
    boxes = torch.cat(boxes, dim=0)
    scores = torch.cat(scores, dim=0)

    # Keep track of already matched boxes
    detected_targets = torch.zeros(label_images.shape[0], device=device, dtype=torch.long)

    num_detections = boxes.shape[0]
    if num_detections == 0:
        return 0, 0

    scores, indices = torch.sort(scores, dim=0, descending=True)
    detection_images = detection_images[indices]

    true_positives = torch.zeros((num_detections), device=device, dtype=torch.float)
    false_positives = torch.zeros((num_detections), device=device, dtype=torch.float)
    iou = torch.zeros((num_detections), device=device, dtype=torch.float)
    for i in range(num_detections):
        box = boxes[i : i + 1]
        image = detection_images[i]

        # Check for boxes from the same image
        image_targets = targets[label_images == image]
        if image_targets.shape[0] == 0:
            false_positives[i] = 1
            continue

        ious = torchvision.ops.box_iou(box, image_targets)
        max_iou, idx = torch.max(ious[0], dim=0)
        # Get the index of the target that we matched with
        original_idx = torch.arange(0, targets.shape[0], dtype=torch.long)[label_images == image][idx]

        # Handle a match - if we've already matched to this target before, count as false positive
        if max_iou > iou_threshold:
            if detected_targets[original_idx] == 0:
                iou[i] = max_iou
                true_positives[i] = 1
                detected_targets[original_idx] = 1
            else:
                false_positives[i] = 1
        else:
            false_positives[i] = 1

    cumulative_true_positives = torch.cumsum(true_positives, dim=0)
    cumulative_false_positives = torch.cumsum(false_positives, dim=0)
    cumulative_precision = cumulative_true_positives / (cumulative_true_positives + cumulative_false_positives + 1e-7)
    cumulative_recall = cumulative_true_positives / targets.shape[0]

    # Calculate precision
    recall_thresholds = torch.arange(start=0, end=1.1, step=.1)
    precisions = torch.zeros((len(recall_thresholds)), device=device, dtype=torch.float)
    for i, t in enumerate(recall_thresholds):
        recalls_above_t = cumulative_recall >= t
        if recalls_above_t.any():
            precisions[i] = cumulative_precision[recalls_above_t].max()
        else:
            precisions[i] = 0.
    average_precisions = precisions.mean()
    mean_iou = iou.mean()
    return average_precisions, mean_iou

def label_on_image(image_file, boxes):

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

    image = skimage.io.imread(image_file)

    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")

    for i in range(N):
        if np.size(boxes[i]) == 0:
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        x1, y1, x2, y2 = boxes[i]   # corners of bounding box. The accuracy of this algorithm could be greatly improved by  interpreting the masks directly as opposed to using the bounding box

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
    if not os.path.isdir("./ID_table/"):
        os.makedirs("./ID_table")
    np.savetxt("./ID_table/" + str(counter) + ".csv", curr, delimiter=",")  # save all data to csv
    prev = curr
    curr = np.zeros((maxBubblesPerFrame, cols))
    counter = counter + 1
# end Resetting

    return image

def run_apply(args):
    test_dir = os.path.join(args.root, args.name, args.test_dir)
    if not (os.path.isdir(os.path.join(test_dir, 'patches')) and os.path.isdir(os.path.join(test_dir, 'targets'))):
        print("Skipping apply, no test data")
        return
    torch.manual_seed(0)

    if args.mp and args.gpu > -1:
        gpu = args.gpu - 1
    else:
        gpu = args.gpu
    if torch.cuda.is_available() and gpu > -1:
        device = torch.device('cuda', gpu)
        model = model_mappings(args)
        model.to(device)
    else:
        model = model_mappings(args)
    test_set = Dataset(test_dir, [], False)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, drop_last=False, collate_fn=collate_fn)

    model_dir = 'saved/%s.pth' % (args.name)#, args.version)
    model.load_state_dict(torch.load(model_dir)['model_state_dict'])

    save_dir = '%s/predictions/%s' % (test_dir, args.name)
    needed_subdirs = ['boxes', 'masks', 'labeled_images', 'scores']
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for subdir in needed_subdirs:
        if not os.path.isdir(os.path.join(save_dir, subdir)):
            os.makedirs(os.path.join(save_dir, subdir))

    loss = evaluate(model, test_loader, args.amp, gpu, save_dir = save_dir, test=True)
    return loss

def run_metrics(args, loss=None):
    if args.mp and args.gpu > -1:
        gpu = args.gpu - 1
    else:
        gpu = args.gpu
    if gpu != -1:
        device = torch.device('cuda', gpu)
    else:
        device = torch.device('cpu')
    test_dir = os.path.join(args.root, args.name, args.test_dir)
    save_dir = '%s/predictions/%s' % (test_dir, args.name)
    boxes = [torch.tensor(np.load(os.path.join(save_dir, 'boxes', file)), device=device) for file in os.listdir(os.path.join(save_dir, 'boxes'))]
    scores = [torch.tensor(np.load(os.path.join(save_dir, 'scores', file)), device=device) for file in os.listdir(os.path.join(save_dir, 'scores'))]
    target_box_files = []
    for file in os.listdir(os.path.join(test_dir, 'targets')):
        if 'boxes' in file:
            target_box_files.append(file)
    target_boxes = [torch.tensor(np.load(os.path.join(test_dir, 'targets', file))) for file in target_box_files]

    mAP, iou = 0, 0#get_metrics(boxes, scores, target_boxes, device, 0.5)

    #mAP = stats[0][0]
    print('--- evaluation result ---')
    print('loss: %.5f, mAP %.5f, IoU %.5f' % (loss, mAP, iou))

    return  loss, iou, mAP
