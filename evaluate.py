import math
import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
from models import model_mappings
from utils import Dataset, VideoDataset, collate_fn, get_circle_coords, tqdm
from visualize import label_image, label_volume


def evaluate(model: Module, valid_loader: DataLoader, amp: bool, gpu: int, save_dir: Optional[str] = None, test: bool = False, apply: bool = True, metrics: bool = True) -> Tuple[float, float, float]:
    '''
    Validation/testing loop. Calculates metrics, and if testing, will write out boxes, masks, and scores.
    Parameters
    ----------
    model: Module
        A torchvision compatible object detector, such as Faster-RCNN or Mask-RCNN
    valid_loader: DataLoader
        A dataloader for the validation or testing data. Does not necessarily require labels.
    amp: bool
        Whether or not to use Accelerated Mixed Precision - lowers memory requirements
    gpu: int
        CUDA ordinal for which GPU to run one
    save_dir: str
        Where to write out saved files. Subdirectories will be created for boxes, masks, and scores
    test: bool
        Whether or not this is a testing run, requiring the saving of files
    apply: bool
        Currently unused.
    metrics: bool
        Currently unused.
    Returns
    -------
    loss, iou, mAP: float
        Scores for each metric, if the dataset has labels and is test is set to false (usually for validation)
    loss: float
        Loss, if test is active
    '''
    if gpu != -1:
        device = torch.device('cuda', gpu)
    else:
        device = torch.device('cpu')

    with torch.no_grad():
        # Set model to evaluate mode (disable batchnorm and dropout, output boxes)
        model.eval()
        total_loss = 0
        num_samples = valid_loader.batch_size * len(valid_loader)
        if num_samples == 0:
            print("WARNING: Validation skipped (likely because the number of validations patches < batch size)")
            if test:
                return 0
            else:
                return (0, 0, 0)
        target_boxes = []
        output_list = []
        target_list = []
        # Check if the dataset has labels. This doesn't apply for VideoDatasets
        if isinstance(valid_loader.dataset, Dataset) and len(valid_loader) > 0:
            _, targets = next(iter((valid_loader)))
        for j, (images, targets) in enumerate(tqdm(valid_loader)):
            images = list(image.to(device) for image in images)
            # If the target exists, push the target tensors to the right device
            has_target = targets[0] is not None
            if has_target:
                targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(images)
                for i, output in enumerate(outputs):
                    if 'masks' in output:
                        output = process_masks(output)
                    # NMS the boxes prior to evaluation, push to CPU for pyCocoTools
                    indices = torchvision.ops.nms(output['boxes'], output['scores'], 0.3)
                    output = {k: v[indices].cpu() for k, v in output.items()}
                    output = remove_concetric_circles(output)
                    coco_output = {
                        'boxes': output['boxes'],
                        'scores': output['scores'],
                        'labels': output['labels']
                    }

                    output_list.append(coco_output)
                    if has_target:
                        target = targets[i]
                        target_list.append(target)
                        target_boxes.append(target['boxes'].to(device))
                        model.train()
                        loss_dict = model(images, targets)
                        loss = sum(loss for loss in loss_dict.values()).item()
                        model.eval()
                        if not math.isfinite(loss) and not test:
                            print('Loss is %s, stopping training' % loss)
                            return 10000, 0, 0
                        total_loss += loss * valid_loader.batch_size
                    # If set to test/apply mode, save out the predicted bounding boxes, masks, and scores
                    if test:
                        if isinstance(valid_loader.dataset, Dataset):
                            this_image = os.path.basename(sorted(os.listdir(valid_loader.dataset.patch_root))[j])
                            write_image = True
                        else:
                            filename_length = len(str(len(valid_loader.dataset)))
                            this_image = str(j).zfill(filename_length)
                            write_image = False
                        if 'boxes' in coco_output:
                            box_cpu = output['boxes'].cpu().numpy()
                            np.save(os.path.join(save_dir, 'boxes', this_image), box_cpu)
                        if 'masks' in coco_output:
                            mask_cpu = coco_output['masks'].cpu().numpy()
                            np.save(os.path.join(save_dir, 'masks', this_image), mask_cpu)
                        if 'scores' in coco_output:
                            scores_cpu = coco_output['scores'].cpu().numpy()
                            np.save(os.path.join(save_dir, 'scores', this_image), scores_cpu)
                        if write_image:
                            if has_target:
                                label_file = os.path.join(valid_loader.dataset.target_root, this_image[:-4] + '_boxes.npy')
                            else:
                                label_file = None
                            label_image(os.path.join(valid_loader.dataset.patch_root, this_image), os.path.join(save_dir, 'boxes', this_image + '.npy'), save_dir, save_image=True, label_file=label_file)
        # Finish pyCocoTools evaluation
        if has_target and not test:
            mAP, iou = get_metrics(output_list, target_list, valid_loader.dataset)
            print('--- evaluation result ---')
            print('loss: %.5f, mAP %.5f, IoU %.5f' % (total_loss / num_samples, mAP, iou))
            return total_loss / num_samples, iou, mAP
        return total_loss / num_samples


def get_metrics(outputs: List[Dict], targets: List[Dict], dataset: Dataset) -> Tuple[float, float]:
    '''
    Calculate metrics (mAP, IoU) for the model outputs using pyCocoTools for mAP and by matches boxes
    for the IoU.
    Parameters
    ----------
    outputs: List
        List of predicted dictionaries for each image, containing the bounding boxes, masks, etc.
    targets: List
        List of target dictionaries (labels) for each image, containing the same keys as outputs
    dataset: Dataset
        A COCO formatted dataset
    Returns
    -------
    mAP, iou: float
        Scores for the model performance
    '''
    coco = get_coco_api_from_dataset(dataset)
    iou_types = ['bbox']
    coco_evaluator = CocoEvaluator(coco, iou_types)

    res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
    coco_evaluator.update(res)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    stats = coco_evaluator.summarize()

    image, _ = dataset[0]
    width, height = F.get_image_size(image)
    ious = []
    for target, output in zip(targets, outputs):
        target_mask = torch.zeros((height, width), dtype=torch.uint8)
        output_mask = torch.zeros((height, width), dtype=torch.uint8)
        for box in target['boxes']:
            x0, y0, x1, y1 = box.cpu().numpy()
            x_dia = x1 - x0
            y_dia = y1 - y0
            avg_rad = (x_dia + y_dia) / 4
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            rr, cc = get_circle_coords(center_y, center_x, avg_rad, avg_rad, height, width)
            target_mask[rr, cc] = 1

        for box in output['boxes']:
            x0, y0, x1, y1 = box.cpu().numpy()
            x_dia = x1 - x0
            y_dia = y1 - y0
            avg_rad = (x_dia + y_dia) / 4
            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2
            rr, cc = get_circle_coords(center_y, center_x, avg_rad, avg_rad, height, width)
            output_mask[rr, cc] = 1

        intersection = torch.sum(target_mask * output_mask)
        union = torch.sum((target_mask + output_mask) > 0)

        ious.append(intersection / union)

    iou = np.mean(ious)

    return stats[0][0], iou


def process_masks(output: Dict[str, torch.Tensor], iou_threshold: float = 0.8) -> Dict[str, torch.Tensor]:
    '''
    Post-processing for object segmentation models. Throws out bounding boxes that do not agree with their masks.
    Parameters:
    output: Dict
        A dictionary for the prediction on a given image
    iou_threshold: float
        The minimum IoU required for a bounding box's inscribed circle and mask to "agree"
    Returns
    -------
    output: Dict
        The input 'output' dict with disagreeing annotations removed
    '''
    indices = []
    for i, (box, mask) in enumerate(zip(output['boxes'], output['masks'])):
        mask = mask[0] >= 0.5
        box_mask = torch.zeros_like(mask)
        x0, y0, x1, y1 = box.cpu().numpy()
        x_dia = x1 - x0
        y_dia = y1 - y0
        avg_rad = (x_dia + y_dia) / 4
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        rr, cc = get_circle_coords(center_y, center_x, avg_rad, avg_rad, mask.shape[0], mask.shape[1])
        box_mask[rr, cc] = 1

        intersection = torch.sum(box_mask * mask)
        union = torch.sum((box_mask + mask) > 0)
        iou = intersection / union

        if iou > iou_threshold:
            indices.append(i)

    output = {k: v[indices] for k, v in output.items()}
    return output


def remove_concetric_circles(output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    '''
    Vectorized helper method to remove concentric boxes.
    Parameters
    ----------
    output: Dict
        The predicted dictionary for a given image
    Returns
    -------
    output: Dict
        The input with concentric boxes thrown out
    '''
    boxes = output['boxes']
    # X, Y, R
    circle_coords = torch.zeros((boxes.shape[0], 3), device=boxes.device, dtype=torch.float32)
    circle_coords[:, 0] = (boxes[:, 2] + boxes[:, 0]) / 2
    circle_coords[:, 1] = (boxes[:, 3] + boxes[:, 1]) / 2
    circle_coords[:, 2] = (boxes[:, 2] + boxes[:, 3] - boxes[:, 0] - boxes[:, 1]) / 4
    grid = torch.tile(circle_coords, (boxes.shape[0], 1)).reshape(boxes.shape[0], boxes.shape[0], 3)

    # Necessary for broadcasting
    x_y_coords = circle_coords[:, np.newaxis, :2]
    # Subtract the row's xy coords from the col xy
    grid[:, :, :2] -= x_y_coords
    center_dist_and_r = torch.empty((boxes.shape[0], boxes.shape[0], 2), dtype=torch.float32, device=boxes.device)
    # Set the first elem of each row - col to be the distance of their centers
    center_dist_and_r[:, :, 0] = torch.sqrt(torch.sum(grid[:, :, :2]**2, dim=-1))
    # Broadcast
    rads = circle_coords[:, np.newaxis, 2:]
    # Set the second elem of each row - col to be col radius - row radius
    center_dist_and_r[:, :, 1:] = grid[:, :, 2:] - rads
    # Center distance - (col rad - row rad)
    diff = center_dist_and_r[:, :, 0] - center_dist_and_r[:, :, 1]
    # N x N: true if row index is contained in col index
    row_contained_in_col = diff <= 0
    # N: Number of bubbles the index is contained in + 1
    contained_in_other = torch.sum(row_contained_in_col, dim=-1)
    indices = contained_in_other <= 1

    output = {k: v[indices] for k, v in output.items()}
    return output


def run_apply(args: SimpleNamespace) -> Optional[float]:
    '''
    Main method for running inference without testing.
    Parameters
    ----------
    args: SimpleNamespace
        Namespace containing all options and hyperparameters
    Returns
    -------
    loss: float
        The loss obtained through evaluation, if labels exist.

    '''
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
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, drop_last=False, collate_fn=collate_fn, pin_memory=True)

    model_dir = 'saved/%s.pth' % (args.name)#, args.version)
    model.load_state_dict(torch.load(model_dir)['model_state_dict'])
    if args.video:
        save_dir = os.path.dirname(args.video) + '/' + os.path.splitext(os.path.basename(args.video))[0] + '/' + args.name
    else:
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


def run_metrics(args: SimpleNamespace, loss: Optional[float] = None):
    '''
    Main method for calculating metrics other than loss on given outputs, writes to a file.
    Parameters
    ----------
    args: SimpleNamespace
        Namespace containing all options and hyperparameters
    loss: float
        Loss calculated from apply, if it exists.
    Returns
    -------
    loss, iou, mAP
        Scores for model performance
    '''
    test_dir = os.path.join(args.root, args.name, args.test_dir)
    save_dir = '%s/predictions/%s' % (test_dir, args.name)
    output_list = []
    files = sorted(os.listdir(os.path.join(save_dir, 'boxes')))
    for file in files:
        box = torch.from_numpy(np.load(os.path.join(save_dir, 'boxes', file)))
        score = torch.from_numpy(np.load(os.path.join(save_dir, 'scores', file)))
        output = {
            'boxes': box,
            'scores': score,
            'labels': torch.ones(box.shape[0], dtype=torch.int64)
        }
        output_list.append(output)
    stats_dir = os.path.join('saved/statistics', os.path.basename(args.root))
    if not os.path.isdir(stats_dir):
        os.makedirs(stats_dir)

    files = sorted(list(set([filename[:-9] for filename in os.listdir(os.path.join(test_dir, 'targets'))])))
    target_list = []
    for i, file in enumerate(files):
        box = torch.from_numpy(np.load(os.path.join(test_dir, 'targets', file + 'boxes.npy')))
        mask = torch.from_numpy(np.load(os.path.join(test_dir, 'targets', file + 'masks.npy')))
        area = torch.from_numpy(np.load(os.path.join(test_dir, 'targets', file + 'areas.npy')))
        target = {
            'boxes': box,
            'masks': mask,
            'areas': area,
            'labels': torch.ones(box.shape[0], dtype=torch.int64),
            'iscrowd': torch.zeros(box.shape[0], dtype=torch.uint8),
            'image_id': torch.tensor([i])
        }
        target_list.append(target)
    dataset = Dataset(os.path.join(args.root, args.name, args.test_dir), [], False)
    mAP, iou = get_metrics(output_list, target_list, dataset)

    if loss is not None:
        print('--- evaluation result ---')
        result_string = 'loss: %.5f, mAP %.5f, IoU %.5f' % (loss, mAP, iou)
    else:
        print('--- evaluation result ---')
        result_string = 'mAP %.5f, IoU %.5f' % (mAP, iou)
    print(result_string)

    with open(os.path.join(stats_dir, '%s.txt' % (args.name)), 'w') as fp:
        fp.write(result_string)

    return loss, iou, mAP
