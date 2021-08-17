import argparse
import json
import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import skimage.draw
import torch
import torch.utils.data
from PIL import Image
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

from transforms import (Compose, RandomApply, RandomHorizontalFlip,
                        RandomRotation, RandomVerticalFlip, ToTensor)


def get_transforms(training: bool, transforms: List[str]) -> Compose:
    composition = []
    composition.append(ToTensor())
    if training and transforms:
        transform_mapping = {
            'horizontal_flip': RandomHorizontalFlip(0.5),
            'vertical_flip': RandomVerticalFlip(0.5),
            'rotation': RandomApply([RandomRotation(80)], p=0),
        }
        for transform in transforms:
            if transform in transform_mapping:
                composition.append(transform_mapping[transform])
            else:
                raise ValueError('Invalid transform %s supplied' % transform)
    return Compose(composition)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transforms: List[str], training: bool = False):
        super(Dataset).__init__()
        self.image_root = os.path.join(root, 'images/')
        self.json_root = os.path.join(root, 'json/')
        self.image_ids = os.listdir(self.image_root)
        self.training = training
        self.transforms = transforms

    def __getitem__(self, index):
        target = {}
        image_id = self.image_ids[index]
        image = Image.open(os.path.join(self.image_root, image_id)).convert(mode='RGB')
        width = image.width
        height = image.height
        # Store the bubbles info
        with open(os.path.join(self.json_root, image_id[:-4] + '.json')) as fp:
            json_data = json.load(fp)
            annotations = json_data['annotations']
            bubbles = []
            for circle in annotations:
                if 'ellipse' in circle:
                    ellipse = circle['ellipse']
                    center_x = ellipse['center']['x']
                    center_y = ellipse['center']['y']
                    radius_x = ellipse['radius']['x']
                    radius_y = ellipse['radius']['y']
                    bubbles.append(torch.tensor([center_x, center_y, radius_x, radius_y]))
        # Allocate space for mask, boxes, and area
        mask = torch.zeros((len(bubbles), height, width), dtype=torch.uint8)
        boxes = torch.empty((len(bubbles), 4), dtype=torch.float32)
        areas = torch.empty(len(bubbles), dtype=torch.float32)
        for i, bubble in enumerate(bubbles):
            center_x = bubble[0]
            center_y = bubble[1]
            radius_x = bubble[2]
            radius_y = bubble[3]
            # Get coordinates for bounding box, making sure to bound by image size
            x_0 = max(center_x - radius_x, 0)
            y_0 = max(center_y - radius_y, 0)
            x_1 = min(center_x + radius_x, width - 1)
            y_1 = min(center_y + radius_y, height - 1)
            # if x_1 < 0:
                # print(center_x)
                # print(center_y)
                # print(radius_x)
                # print(radius_y)
            # Save box and area
            boxes[i, :] = torch.tensor([x_0, y_0, x_1, y_1], dtype=torch.float32)
            areas[i] = (x_1 - x_0) * (y_1 - y_0)
            # Generate points for mask
            rr, cc = skimage.draw.ellipse(center_y, center_x, radius_y, radius_x)
            # Limit these to only points in the image
            cc = cc[rr < height]
            rr = rr[rr < height]
            rr = rr[cc < width]
            cc = cc[cc < width]
            cc = cc[rr >= 0]
            rr = rr[rr >= 0]
            rr = rr[cc >= 0]
            cc = cc[cc >= 0]
            mask[i, rr, cc] = 1
        # Fill target according to COCO standard
        target['boxes'] = boxes
        target['area'] = areas
        target['labels'] = torch.ones(boxes.shape[0], dtype=torch.int64)
        target['iscrowd'] = torch.zeros(boxes.shape[0], dtype=torch.uint8)
        target['masks'] = mask
        target['image_id'] = torch.tensor([index])
        transform = get_transforms(self.training, self.transforms)
        image, target = transform(image, target)
        return (image, target)

    def __len__(self):
        return len(self.image_ids)

def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloaders(args: SimpleNamespace, world_size: Optional[int], rank: Optional[int]) -> Tuple[DataLoader, DataLoader]:
    training_data = Dataset(os.path.join(args.root, 'train'), args.transforms, True)
    validation_data = Dataset(os.path.join(args.root, 'validation'), args.transforms, False)

    if world_size is None or rank is None:
        train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        valid_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        return train_loader, valid_loader

    train_sampler = DistributedSampler(training_data, num_replicas=world_size, rank=rank, drop_last=True)
    valid_sampler = DistributedSampler(validation_data, num_replicas=world_size, rank=rank, drop_last=True)

    train_loader = DataLoader(training_data, batch_size=args.batch_size, collate_fn=collate_fn, drop_last=True, sampler=train_sampler)
    valid_loader = DataLoader(validation_data, batch_size=args.batch_size, collate_fn=collate_fn, drop_last=True, sampler=valid_sampler)

    return train_loader, valid_loader

def warmup_lr_scheduler(optimizer: Optimizer, warmup_iters: int, warmup_factor: float) -> torch.optim.lr_scheduler.LambdaLR:
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def gen_args(args: argparse.Namespace, defaults: Dict, file_args: Optional[Dict] = None, config: Optional[str] = None) -> SimpleNamespace:
    full_args = defaults.copy()
    if file_args is not None:
        for key, value in file_args.items():
                full_args[key] = value
    for key in full_args.keys(): #Sub in with direct from command args
        if hasattr(args, key) and getattr(args, key) is not None:
            new_attr = getattr(args, key)
            if type(new_attr) is not bool or new_attr is not defaults[key]:
                full_args[key] = new_attr
    if config:
        full_args['config'] = config
    return SimpleNamespace(**full_args)
