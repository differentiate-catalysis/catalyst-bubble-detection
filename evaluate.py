import math
import sys
from typing import List, Tuple
import os
from utils import VideoDataset, get_iou_types

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
        # Check if the dataset has labels. This doesn't apply for VideoDatasets
        if isinstance(valid_loader.dataset, Dataset):
            _, targets = next(iter((valid_loader)))
            if targets[0] is not None:
                coco = get_coco_api_from_dataset(valid_loader.dataset)
                iou_types = get_iou_types(model)
                coco_evaluator = CocoEvaluator(coco, iou_types)
        for j, (images, targets) in enumerate(tqdm(valid_loader)):
            images = list(image.to(device) for image in images)
            # If the target exists, push the target tensors to the right device
            has_target = targets[0] is not None
            if has_target:
                targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(images)
                res = {}
                for i, output in enumerate(outputs):
                    # NMS the boxes prior to evaluation, push to CPU for pyCocoTools
                    indices = torchvision.ops.nms(output['boxes'], output['scores'], 0.3)
                    output = {k: v[indices].cpu() for k, v in output.items()}
                    boxes.append(output['boxes'].to(device))
                    scores.append(output['scores'].to(device))
                    if has_target:
                        target = targets[i]
                        target_boxes.append(target['boxes'].to(device))
                        res[target['image_id'].item()] = output
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
                    # If set to test/apply mode, save out the predicted bounding boxes, masks, and scores
                    if test:
                        if isinstance(valid_loader.dataset, Dataset):
                            this_image = os.path.basename(os.listdir(valid_loader.dataset.patch_root)[j])
                        else:
                            filename_length = len(str(len(valid_loader.dataset)))
                            this_image = str(j).zfill(filename_length)
                        if 'boxes' in output:
                            box_cpu = output['boxes'].cpu().numpy()
                            np.save(os.path.join(save_dir, 'boxes', this_image), box_cpu)
                        if 'masks' in output:
                            mask_cpu = output['masks'].cpu().numpy()
                            np.save(os.path.join(save_dir, 'masks', this_image), mask_cpu)
                        if scores:
                            scores_cpu = output['scores'].cpu().numpy()
                            np.save(os.path.join(save_dir, 'scores', this_image), scores_cpu)
        # Finish pyCocoTools evaluation
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

def check_coalesce(center,radius,prev_circles,curr_circles,min_c=10):
    '''
    Performs a check of coalescence of bubbles between frames. Definition of coalescence:
    -Take two bubbles A and B at time (T-1) with radii r_A, r_B, respectively.
    -The two bubbles must be at a distance (r_A+r_B+min_c), otherwise coalescence not considered
    -At time step T, only one of bubbles A and B remain, and the radius of A has increased substantially.
    Parameters
    ----------
    center: list
        coordinates (x,y) for the center of a circle
    radius: float (or int)
        radius of circle in question
    prev_circles: dict
        dictionary containing all relevant info of past circles
    curr_circles: dict
        dictionary containing all relevant info of current circles
    min_c: int
        minimum distance between neighboring circles at which coalescence is deemed possible
    Returns
    -------
    is_coalesced: bool
        boolean where if True coalesence occurred, else False
    '''
    past_centers = np.array([np.array(prev_circles[i]['center']) for i in sorted(prev_circles.keys())])
    all_new_centers = np.array([np.array(curr_circles[i]['center']) for i in sorted(curr_circles.keys())])
    past_ind = np.argmin(np.sum((past_centers-np.array(center))**2,axis=1)**0.5)
    past_match = prev_circles[past_ind]
    poss_coalesce_inds = []
    for i in sorted(prev_circles.keys()):
        if i==past_ind: #Skip duplicates for past bubble frame
            continue
        poss_center = np.array(prev_circles[i]['center'])
        curr_center = past_match['center']
        if np.sum((poss_center-curr_center)**2)**0.5 < prev_circles[i]['radius']+past_match['radius']+min_c:
            poss_coalesce_inds.append(i) #Circles are close enough that they have potential to merge
    #For potential merge candidates, did any of them actually coalesce? If no match, then coalescenced occurred
    for i in poss_coalesce_inds:
        poss_center = np.array(prev_circles[i]['center'])
        new_center_dists = np.sum((all_new_centers-np.array(poss_center))**2,axis=1)**0.5
        if min(new_center_dists)>10: #No match to previous frame, so coalescence must have occurred
            return True #Coalescence
    return False #No coalescence


def compare_bubbles(center,radius,prev_circles,min_dist=10):
    '''
    Takes an existing bubble in a frame, determines whether the bubble is new, assessess volume change
    Parameters
    ----------
    center: list
        coordinates (x,y) for the center of a circle
    radius: float (or int)
        radius of circle in question
    prev_circles: dict
        dictionary containing all relevant info of past circles
    min_dist: int
        the minimum distance between circle centers at which one judges the formation of a new bubble
    Returns
    -------
    new: bool
        boolean stating whether a bubble is new in the current frame (True) or recurring (False)
    coalesced: bool (or None)
        boolean stating whether a bubble coalesced between frames (None if undetermined)
    volume_change: float (or None)
        change in volume of a bubble between slides (units are technically in px**3)
    old_id: int (or None)
        if bubble is not new, then keep the same old id between frames
    '''
    all_centers = np.array([np.array(prev_circles[i]['center']) for i in sorted(prev_circles.keys())])
    if len(all_centers)==0:
        new=True; coalesced=False; volume_change=None; old_id=None
        return new, coalesced, volume_change, old_id
    #Find matching circle based on min_dist parameter
    center_dists = np.sum((all_centers-np.array(center))**2,axis=1)**0.5
    if min(center_dists)<=10: #match located in previous frame
        matching_circle_data = prev_circles[np.argmin(center_dists)]
        new=False
        volume_change = 2.*np.pi / 3 * (radius**3 - matching_circle_data['radius']**3)
        if volume_change<=10: #Can edit threshold on what sort of volume change warrants coalescence
            coalesced=False
            return new, coalesced, volume_change, matching_circle_data['id']
        else:
            coalesced=None
            return new, coalesced, volume_change, matching_circle_data['id'] 
    else: #No match found, therefore circle is new
        new=True; coalesced=False; volume_change=None; old_id=None
        return new, coalesced, volume_change, old_id

def redraw_fig(frame,bubble_dict,color_dict={'new':(224, 230, 73),'coalesced':(96, 66, 245),\
                  'growing':(50, 168, 82),'shrinking':(150, 0, 0),'unchanged':(179, 247, 255)}):
    '''
    Uses OpenCV functions on an existing frame to add circles based on category as well as id numbers
    Parameters
    ----------
    frame: ndarray
        image array of shape (x_len, y_len, 3) containing bubbles
    bubble_dict: dict
        dictionary containing all relevant info of current circles
    color_dict: dict
        dictionary containing all RGB colors for five modes (new, coalesced, growing, shrinking, unchanged)
    Returns
    -------
    edit_frame: ndarray
        image array after cv2 has inserted circles and text at circle locations
    '''
    change_cutoff=50
    edit_frame = frame.copy()
    for b in bubble_dict.keys():
        bubble = bubble_dict[b]
        color=None
        if bubble['new']: color = color_dict['new']
        if bubble['coalesced']: color = color_dict['coalesced']
        if color is None: #New and coalesced categories take precedent over others
            if bubble['volume_change'] > change_cutoff: color = color_dict['growing']
            elif bubble['volume_change'] < -change_cutoff: color = color_dict['shrinking']
            else: color = color_dict['unchanged']
        cv2.circle(edit_frame, (bubble['center_x'], bubble['center_y']), int(bubble['radius']), color, 3)
        cv2.putText(edit_frame, str(bubble['id']), (bubble['center_x']-9, bubble['center_y']+7),\
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return edit_frame

def label_image(image_file, npy_file, bubble_color=(255,0,0) write_frame_csv=False, save_image=False):
    '''
    Takes in an image file a npy file with bounding boxes, returns a single image with bubbles marked.
    Parameters
    ----------
    image_file: str
        name of image file to annotate
    npy_file: str
        name of .npy file containing bounding boxes corresponding to the image frame
    bubble_color: tuple
        RGB color used to label bubbles in the image
    write_frame_csv: bool
        trigger to write a single csv file that describe all bubbles
    save_image: bool
        trigger to write a separate image file with circles and ids per bubble
    Returns
    -------
    edit_frame: ndarray
        final image after insertion of circles at marked bubble coordinates
    '''
    color_dict = {}
    for i in ['new','coalesced','growing','shrinking','unchanged']: color_dict[i]=bubble_color
    full_frame = cv2.imread(image_file)
    curr_boxes = np.load(npy_file)
    curr_circles={} #Tracking dict for all bubbles in frame
    circle_counter = 0
    for box in curr_boxes:
        curr_circles[circle_counter]={} #Tracking dict for current bubble
        center = [0.5*(box[0]+box[2]),0.5*(box[1]+box[3])] #Center averaging
        curr_circles[circle_counter]['center'] = center
        curr_circles[circle_counter]['center_x'] = int(center[0])
        curr_circles[circle_counter]['center_y'] = int(center[1])
        radius = 0.5*(abs(box[0]-center[0]) + abs(box[1]-center[1])) #Radius averaging
        curr_circles[circle_counter]['radius'] = radius
        curr_circles[circle_counter]['diameter'] = int(radius*2)
        curr_circles[circle_counter]['volume_change']=None
        curr_circles[circle_counter]['new']=True
        curr_circles[circle_counter]['coalesced']=False
        curr_circles[circle_counter]['id']=circle_counter
        circle_counter+=1
    if write_frame_csv:
        curr_df = pd.DataFrame.from_dict(curr_circles).T.drop(columns=['center',\
                                                    'radius']).sort_values(by='id')
        curr_df.to_csv(f'frame_single.csv',index=False)
    edit_frame = redraw_fig(full_frame,curr_circles,color_dict=color_dict)
    if save_image:
        cv2.imwrite('frame_single.png',edit_frame)
    return edit_frame


def label_volume(movie_file, npy_folder, write_frame_csv=False, write_overall=False, save_video=False):
    '''
    Takes in a movie file and all specified bounding boxes, returns descriptor files and edited movie.
    Parameters
    ----------
    movie_file: str
        path to the movie file to be parsed
    npy_folder: str
        folder name containing .npy files per frame of bounding boxes
    write_frame_csv: bool
        trigger to write csv files at each frame that describe all bubbles
    write_overall: bool
        trigger to write a csv file that describes the overall movement and process of the bubbles
    save_video: bool
        trigger to write a separate video file with edited frames with circles and ids per bubble
    Returns
    -------
    overall_df: pd.DataFrame
        dataframe describing the full process of bubbles changing between frames
    '''
    capture = cv2.VideoCapture(movie_file)
    new_vid = None
    success = True
    prev_circles = {}
    total_volume_li = []
    num_bubbles_li = []
    max_id_num = 0
    frame = 0
    npy_files = sorted(np.array(os.listdir(npy_folder)).tolist(),\
                       key = lambda x: int(x.split('_')[0]))
    while success: #Iterating over initial image frames, writing new frames to new video
        success,curr_frame = capture.read()
        if new_vid is None and save_video: #Open new video to begin writing, if desired
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            new_vid = cv2.VideoWriter('test_vid_out.mov',fourcc,60,\
                                       (int(capture.get(3)),int(capture.get(4))))
        if not success:
            break
        curr_boxes = np.load(os.path.join(npy_folder,npy_files[frame]))
        curr_circles = {} #Tracking dict for bubbles in current frame
        circle_counter = 0
        vol_in_frame = 0
        coalesce_check = []
        for box in curr_boxes:
            curr_circles[circle_counter]={} #Tracking dict for current bubble
            center = [0.5*(box[0]+box[2]),0.5*(box[1]+box[3])] #Center averaging
            curr_circles[circle_counter]['center'] = center
            curr_circles[circle_counter]['center_x'] = int(center[0])
            curr_circles[circle_counter]['center_y'] = int(center[1])
            radius = 0.5*(abs(box[0]-center[0]) + abs(box[1]-center[1])) #Radius averaging
            curr_circles[circle_counter]['radius'] = radius
            curr_circles[circle_counter]['diameter'] = int(radius*2)
            new, coalesced, volume_change, old_id = compare_bubbles(center,radius,prev_circles)
            if new:
                curr_circles[circle_counter]['id']=max_id_num
                max_id_num+=1
            else:
                curr_circles[circle_counter]['id']=old_id
            curr_circles[circle_counter]['new']=new
            curr_circles[circle_counter]['coalesced']=coalesced
            curr_circles[circle_counter]['volume_change']=volume_change
            vol_in_frame+=2.*np.pi/3. * radius**3 #Sum all circle volumes
            if coalesced is None:
                coalesce_check.append(circle_counter)
            circle_counter+=1
        #Check after for possible coalescences
        for i in coalesce_check:
            curr_circles[i]['coalesced']=check_coalesce(curr_circles[i]['center'],\
                                                        curr_circles[i]['radius'],\
                                                        curr_circles,prev_circles)
        total_volume_li.append(vol_in_frame) #Keep a list of all recorded volumes
        num_bubbles_li.append(int(len(curr_circles)))
        if len(curr_circles)>0:
            curr_df = pd.DataFrame.from_dict(curr_circles).T.drop(columns=['center',\
                                                                'radius']).sort_values(by='id')
        if save_video: #Write edited frame to new video
            new_vid.write(redraw_fig(curr_frame,curr_circles))
        if write_frame_csv:
            curr_df.to_csv(f'frame_{frame}.csv',index=False)
        prev_circles = curr_circles
        frame+=1
    #Release videos when runs are done
    if save_video: new_vid.release()
    capture.release()
    overall_df = pd.DataFrame(np.array([total_volume_li,np.cumsum(total_volume_li).tolist(),\
                                        np.diff(total_volume_li,prepend=0.),num_bubbles_li]).T,\
                              columns=['frame_bubble_V','diff_bubble_V','total_bubble_V','num_bubbles'])
    if write_overall:
        overall_df.to_csv('overall_results.csv',index=False)
    return overall_df

#Test use, if desired
#_ = label_volume('test_bubbles.mov',boxes_npy_folder,write_overall=True,save_video=True)
#_ = label_image('test_image.png',boxes_npy_file)


def run_apply(args):
    if args.video:
        test_dir = os.path.join(os.path.join(args.root, args.name, args.test_dir))
        os.makedirs(os.path.join(test_dir, 'patches'), exist_ok=True)
        os.makedirs(os.path.join(test_dir, 'targets'), exist_ok=True)
    else:
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
    if args.video:
        test_set = VideoDataset(args.video)
    else:
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
