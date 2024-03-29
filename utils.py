import argparse
import json
import os
import pickle
import time
from math import ceil
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import skimage.draw
import torch
import torch.distributed
import torch.utils.data
import torchvision
from PIL import Image
from ray import tune
from torch.nn.modules.module import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from tqdm import tqdm as tqdm_wrapper

from transforms import Compose, ToTensor, CLAHE, transform_mappings


def compute_mean_and_std(split_paths: List[str], image_size: Tuple[int, int]) -> Tuple[List[float], List[float]]:
    '''
    Compute the means and standard deivations of each channel, writes to a stats file.
    Parameters
    ----------
    split_paths: List[str]
        List of paths to search for images under. Typically we'd search under train and validation.
    image_size: Tuple[int, int]
        Size of each image in pixels.
    Returns
    -------
    means, std
        Means and standard deivations of the dataset
    '''
    images = []
    for split_path in split_paths:
        images.extend(list(Path(split_path).rglob('*.jpg')))
        images.extend(list(Path(split_path).rglob('*.png')))
    mean_tensor = torch.zeros((3, len(images)))
    std_tensor = torch.zeros((3, len(images)))
    for i, image_path in enumerate(tqdm(images, desc='Calculating mean...')):
        im = Image.open(image_path).convert('RGB')
        im = pil_to_tensor(im).float() / 255
        means = torch.mean(im, dim=(-1, -2))
        mean_tensor[:, i] = means
    means = torch.mean(mean_tensor, dim=-1)
    for i, image_path in enumerate(tqdm(images, desc='Calculating stddev...')):
        im = Image.open(image_path).convert('RGB')
        im = pil_to_tensor(im).float() / 255
        stds = torch.mean((im - means.reshape(3, 1, 1))**2, dim=(-1, -2))
        std_tensor[:, i] = stds
    weight = image_size[0] * image_size[1]
    std = torch.sqrt(weight * torch.sum(std_tensor, dim=-1) / (weight * len(images) - 1))
    means = means.cpu().tolist()
    std = std.cpu().tolist()
    with open(os.path.join(split_paths[0], '..', 'stats.json'), 'w') as f:
        json.dump({'mean': means, 'std': std}, f)

    return means, std


def compute_mean_and_std_video(data_root: str, video_file: str, image_size: Tuple[int, int]) -> Tuple[List[float], List[float]]:
    '''
    Compute the means and standard deivations of each channel, writes to a stats file.
    Parameters
    ----------
    data_root: str
        Where to write the stats file to
    video_file: str
        The video file to compute stats over
    image_size: Tuple[int, int]
        Size of each frame in pixels.
    Returns
    -------
    means, std
        Means and standard deivations of the dataset
    '''
    dataset = VideoDataset(video_file)
    mean_tensor = torch.zeros((3, len(dataset)))
    std_tensor = torch.zeros((3, len(dataset)))
    for i, (im, _) in enumerate(tqdm(dataset, desc='Calculating mean...')):
        im = im.float() / 255
        means = torch.mean(im, dim=(-1, -2))
        mean_tensor[:, i] = means
    means = torch.mean(mean_tensor, dim=-1)
    dataset = VideoDataset(video_file)
    for i, (im, _) in enumerate(tqdm(dataset, desc='Calculating stddev...')):
        im = im.float() / 255
        stds = torch.mean((im - means.reshape(3, 1, 1))**2, dim=(-1, -2))
        std_tensor[:, i] = stds
    weight = image_size[0] * image_size[1]
    std = torch.sqrt(weight * torch.sum(std_tensor, dim=-1) / (weight * len(dataset) - 1))
    means = means.cpu().tolist()
    std = std.cpu().tolist()
    with open(os.path.join(data_root, 'stats.json'), 'w') as f:
        json.dump({'mean': means, 'std': std}, f)

    return means, std


def get_transforms(training: bool, transforms: List[str]) -> Compose:
    composition = []
    composition.append(ToTensor())  # When gpu == -1, this will run on CPU.
    # composition.append(CLAHE())
    # composition.append(AutoExpose())
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
        self.image_ids = sorted(os.listdir(self.patch_root))
        self.training = training
        self.stats = os.path.join(root, '..', 'stats.json')
        self.transform = get_transforms(self.training, transforms)
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

        if not (torch.any(mask) or torch.any(boxes) or torch.any(areas)): #Account for 0 targets (i.e. unlabeled image evaluated)
            target = None
            image, _ = self.transform(image)
        else:
            target['boxes'] = boxes
            target['area'] = areas
            target['labels'] = torch.ones(boxes.shape[0], dtype=torch.int64)
            target['iscrowd'] = torch.zeros(boxes.shape[0], dtype=torch.uint8)
            target['masks'] = mask
            target['image_id'] = torch.tensor([index])
            # Augment the image and target
            start = time.time()
            image, target = self.transform(image, target)
            end = time.time()
            # print('Augmenting took %0.4f ms' % ((end - start)*1000))
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


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_file: str):
        super(VideoDataset).__init__()
        torchvision.set_video_backend('video_reader')
        self.reader = torchvision.io.VideoReader(video_file)
        metadata = self.reader.get_metadata()
        self.transform = get_transforms(True, ['clahe'])
        self.length = ceil(metadata['video']['duration'][0] * metadata['video']['fps'][0])
        self.overfill = False

    def __getitem__(self, index):
        target = None
        # start = time.time()
        if not self.overfill:
            try:
                image = next(self.reader)['data']
            except StopIteration:
                self.reader.seek(0)
                image = next(self.reader)['data']
                self.overfill = True
        image = to_pil_image(image, mode='RGB')
        if self.overfill:
            image = torch.zeros(image.shape)
        image, _ = self.transform(image)
        # print('Dataloading took %f seconds' % (time.time() - start))
        if index == self.length - 1:
            self.reader.seek(0)
        return image, target

    def __len__(self):
        return self.length


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloaders(args: SimpleNamespace, world_size: Optional[int], rank: Optional[int]) -> Tuple[DataLoader, DataLoader]:
    '''
    Instantiates dataloaders for training and validation.
    Parameters
    ----------
    args: SimpleNamespace
        Namespace containing all options and hyperparameters
    world_size: Optional[int]
        If multiprocessing, set as number of nodes * number of GPUs. If not, set to None
    rank: Optional[int]
        If multiprocessing, set as the rank of the process. If not, set to None
    Returns
    -------
    train_loader, valid_loader
        Dataloaders for training and validation
    '''
    training_data = Dataset(os.path.join(args.root, args.name, 'train'), args.transforms, True, num_images=args.num_images)
    print("Number of labels in training data: " + str(training_data.get_num_labels()))
    validation_data = Dataset(os.path.join(args.root, args.name, 'validation'), args.transforms, False)

    if world_size is None or rank is None:
        train_loader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True, num_workers=args.data_workers, pin_memory=True)
        valid_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, drop_last=True, num_workers=args.data_workers, pin_memory=True)
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


def gen_args(args: Union[SimpleNamespace, argparse.Namespace], defaults: Dict, file_args: Optional[Dict] = None, config: Optional[str] = None) -> Tuple[SimpleNamespace, List[str], List[str]]:
    full_args = defaults.copy()
    changed = []
    explicit = []
    if file_args is not None:
        for key, value in file_args.items():
                full_args[key] = value
                changed.append(key)
    for key in full_args.keys(): #Sub in with direct from command args
        if hasattr(args, key) and getattr(args, key) is not None:
            new_attr = getattr(args, key)
            if type(new_attr) is not bool or new_attr is not defaults[key]:
                full_args[key] = new_attr
                changed.append(key)
                explicit.append(key)
    if config:
        full_args['config'] = config
    return SimpleNamespace(**full_args), changed, explicit


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
    Copied from torchvision code. Run all_gather on arbitrary picklable data (not necessarily tensors)
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


def get_circle_coords(center_y: float, center_x: float, rad_y: float, rad_x: float, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Generates a list of coordinates inside of a circle.
    Parameters
    ----------
    center_y: float
        Y value of the center pixel of the circle
    center_x: float
        X value of the center pixel of the circle
    rad_y: float
        Y radius (technically for ellipses)
    rad_x: float
        X radius (technically for ellipses)
    height: int
        Height of the image in pixels
    width: int
        Width of the image in pixels
    Returns
    -------
    rr, cc
        Row and column indices of coordinates inside the circle
    '''
    rr, cc = skimage.draw.ellipse(center_y, center_x, rad_y, rad_x)
    cc = cc[rr < height]
    rr = rr[rr < height]
    rr = rr[cc < width]
    cc = cc[cc < width]
    cc = cc[rr >= 0]
    rr = rr[rr >= 0]
    rr = rr[cc >= 0]
    cc = cc[cc >= 0]
    return rr, cc


def tqdm(iterable: Iterable, **kwargs):
    if tune.is_session_enabled():
        return iterable
    return tqdm_wrapper(iterable, **kwargs)
