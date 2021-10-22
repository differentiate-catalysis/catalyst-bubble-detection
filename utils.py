import argparse
import json
import os
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import skimage.draw
import torch
import numpy as np
from torch.nn.modules.module import Module
import torch.utils.data
from PIL import Image
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
from torchvision.transforms.functional import to_pil_image

from transforms import Compose, ToTensor, transform_mappings
import pickle


def get_transforms(training: bool, transforms: List[str]) -> Compose:
    composition = []
    composition.append(ToTensor())
    if training and transforms:
        for transform in transforms:
            if transform in transform_mappings:
                composition.append(transform_mappings[transform])
            else:
                raise ValueError('Invalid transform %s supplied' % transform)
    return Compose(composition)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transforms: List[str], training: bool = False, num_images: int = -1):
        super(Dataset).__init__()
        # self.image_root = os.path.join(root, '..', 'images/')
        self.patch_root = os.path.join(root, 'patches/')
        # self.json_root = os.path.join(root, '..', 'json/')
        self.target_root = os.path.join(root, 'targets/')
        self.image_ids = os.listdir(self.patch_root)
        self.training = training
        self.transforms = transforms
        # if num_images == -1:
            # self.num_images = len(os.listdir(self.image_root))
        # else:
            # self.num_images = num_images

    def __getitem__(self, index):
        target = {}
        image_id = self.image_ids[index]
        image = Image.open(os.path.join(self.patch_root, image_id)).convert(mode='RGB')
        mask = torch.from_numpy(np.load(os.path.join(self.target_root, image_id[:-4] + '_masks.npy')))
        boxes = torch.from_numpy(np.load(os.path.join(self.target_root, image_id[:-4] + '_boxes.npy')))
        areas = torch.from_numpy(np.load(os.path.join(self.target_root, image_id[:-4] + '_areas.npy')))
        # masks = torch.sum(mask, dim=0).byte()
        # masks[masks >= 1] = 255
        # img = to_pil_image(masks).convert('L')
        # img.save('/home/jim/masks/%s.png' % image_id)

        # Fill target according to COCO standard
        target['boxes'] = boxes
        target['area'] = areas
        target['labels'] = torch.ones(boxes.shape[0], dtype=torch.int64)
        target['iscrowd'] = torch.zeros(boxes.shape[0], dtype=torch.uint8)
        target['masks'] = mask
        target['image_id'] = torch.tensor([index])
        # Augment the image and target
        transform = get_transforms(self.training, self.transforms)
        image, target = transform(image, target)
        return image, target

    def __len__(self):
        return len(self.image_ids)

    def get_num_labels(self):
        sum = 0
        # for index in range(self.num_images):
            # image_id = os.listdir(self.image_root)[index]
            # with open(os.path.join(self.json_root, image_id[:-4] + '.json')) as fp:
                # json_data = json.load(fp)
                # annotations = json_data['annotations']
                # sum += len(annotations)
        return sum

def collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloaders(args: SimpleNamespace, world_size: Optional[int], rank: Optional[int]) -> Tuple[DataLoader, DataLoader]:
    training_data = Dataset(os.path.join(args.root, args.name, 'train'), args.transforms, True, num_images=args.num_images)
    print("Number of labels in training data: " + str(training_data.get_num_labels()))
    validation_data = Dataset(os.path.join(args.root, args.name, 'validation'), args.transforms, False)

    if world_size is None or rank is None:
        train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=args.jobs)
        valid_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True, num_workers=args.jobs)
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

def gen_args(args: Union[SimpleNamespace, argparse.Namespace], defaults: Dict, file_args: Optional[Dict] = None, config: Optional[str] = None) -> SimpleNamespace:
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

def get_iou_types(model: Module):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    # world_size = get_world_size()
    world_size = 1
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    torch.distributed.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    torch.distributed.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list
