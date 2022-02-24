import math
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.utils.data
import torchvision
from ray import tune
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from models import model_mappings
from utils import Dataset, VideoDataset, collate_fn, get_iou_types
from visualize import label_volume


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
    if args.video:
        npy_dir = os.path.join(save_dir, 'boxes')
        label_volume(args.video, npy_dir, save_dir, True, True, True)
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
    stats_dir = os.path.join('saved/statistics', os.path.basename(args.root))
    if not os.path.isdir(stats_dir):
        os.makedirs(stats_dir)

    mAP, iou = 0, 0#get_metrics(boxes, scores, target_boxes, device, 0.5)

    #mAP = stats[0][0]

    result_string = 'loss: %.5f, mAP %.5f, IoU %.5f' % (loss, mAP, iou)

    print('--- evaluation result ---')
    print(result_string)
    
    with open(os.path.join(stats_dir, '%s.txt' % (args.name)), 'w') as fp:
        fp.write(result_string)

    return  loss, iou, mAP
