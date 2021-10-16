import math
import sys
from typing import List, Tuple
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

from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset


def evaluate(model: Module, valid_loader: DataLoader, amp: bool, gpu: int) -> Tuple[float, float, float]:
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
        coco = get_coco_api_from_dataset(valid_loader.dataset)
        iou_types = get_iou_types(model)
        coco_evaluator = CocoEvaluator(coco, iou_types)
        for j, (images, targets) in enumerate(tqdm(valid_loader)):
            images = list(image.to(device) for image in images)
            targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(images)
                outputs_cpu = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
                res = {target["image_id"].item(): output for target, output in zip(targets, outputs_cpu)}
                coco_evaluator.update(res)
                # print(outputs.shape)
                for i in range(len(targets)):
                    output = outputs[i]
                    target = targets[i]
                    # NMS the boxes
                    nms_boxes, nms_scores = nms(output['boxes'], output['scores'], 0.5)
                    boxes.append(nms_boxes)
                    target_boxes.append(target['boxes'])
                    scores.append(nms_scores)

                    if 'masks' in output:
                        masks = output['masks']
                        masks = torch.sum(masks, dim=0).byte()
                        masks[masks >= 1] = 255
                        img = to_pil_image(masks).convert('L')
                        img.save('/home/jim/outmasks/%d.png' % (j * valid_loader.batch_size + i))
                    # print(outputs)
                    # iou = torchvision.ops.box_iou(output['boxes'], target['boxes'])
                    # dims = iou.shape
                    # matrix = torch.zeros((max(dims[0], dims[1]), max(dims[0], dims[1])))
                    # matrix[:dims[0], :dims[1]] = iou
                    # total_iou += torch.sum(torch.diagonal(matrix)).item()
                # Must set model to train mode to get loss
                model.train()
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values()).item()
                model.eval()
            if not math.isfinite(loss):
                print('Loss is %s, stopping training' % loss)
                if tune.is_session_enabled():
                    tune.report(loss=10000, iou=0, map=0)
                sys.exit(1)
            total_loss += loss * valid_loader.batch_size
        coco_evaluator.synchronize_between_processes()
        coco_evaluator.accumulate()
        stats = coco_evaluator.summarize()
        # print(stats)
        mAP, iou = get_metrics(boxes, scores, target_boxes, device, 0.5)
        mAP = stats[0][0]
        print('--- evaluation result ---')
        print('loss: %.5f, mAP %.5f, IoU %.5f' % (total_loss / num_samples, mAP, iou))

    return total_loss / num_samples, iou, mAP

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
