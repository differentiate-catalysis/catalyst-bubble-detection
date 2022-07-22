import os
import numpy as np

import torch
import torchvision.transforms as T
from torch.utils import data
from torch.utils.data import DataLoader, DistributedSampler
from statistics import mean
from tqdm import tqdm
from types import SimpleNamespace

class AverageMeter(object):
    """
        Computes and stores the average and current value
        Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: torch.Tensor, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Recorder(object):
    def __init__(self, names):
        self.names = names
        self.record = {}
        for name in self.names:
            self.record[name] = []

    def update(self, vals):
        for name, val in zip(self.names, vals):
            self.record[name].append(val)

    def load(self, checkpoint):
        self.record = checkpoint

class Dataset(data.Dataset):
    def __init__(self, root, training, aug, gpu, transforms):
        self.vol_root = root + '/volumes_np/'
        self.label_root = root + '/labels_np/'
        self.vol_names = os.listdir(self.vol_root)
        self.training = training
        self.aug = aug
        self.avg, self.std = average(self.vol_root)
        if gpu == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda', gpu)
        #self.device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
        self.transforms = transforms

    def __getitem__(self, index):
        vol_name = self.vol_names[index]
        vol_dir = self.vol_root + vol_name
        label_dir = '%s%s.npy' % (self.label_root, vol_name[:-4])
        vol = torch.Tensor(np.load(vol_dir)).to(self.device)
        seed = np.random.randint(2147483647)
        vol = transform(vol, self.training, self.aug, seed, self.avg, self.std, transformations=self.transforms)
        try:
            label = torch.Tensor(np.load(label_dir)).to(self.device)
            label = transform(label, self.training, self.aug, seed, self.avg, self.std, transformations=self.transforms)
        except FileNotFoundError:
            label = None
        return vol, label, vol_name

    def __len__(self):
        return len(self.vol_names)

def average(root_dir):
    # Initialize empty lists which will contain average and std. dev. of each image tensor.
    avg_list = []
    std_list = []

    # Loop through all images in image directory defined above.
    for slice in tqdm(os.listdir(root_dir), desc='Normalizing....', unit=' images'):
        img = np.load(os.path.join(root_dir, slice))
        tensor = torch.Tensor(img)
        avg_list.append(torch.mean(tensor).tolist())
        std_list.append(torch.std(tensor).tolist())

    # Returns list with three equal entries for each channel of the RGB image.
    avg = round(mean(avg_list), 3)
    std = round(mean(std_list), 3)
    return avg, std

def transform(tensor, training, aug, seed, avg, std, transformations=['rotation', 'horizontal_flip', 'vertical_flip']):
    '''
        Transforms input volumes
        tensor: a volume tensor of the shape [channels, slices, width, height] or label [slices, width, height]
        training: true if training, false if evaluating
        aug: whether or not to augment images
    '''
    volume_transformation_directory = {
        'rotation': T.RandomRotation(45),
        'horizontal_flip': T.RandomHorizontalFlip(),
        'vertical_flip': T.RandomVerticalFlip(),
        'depth_flip': T.RandomApply([T.Lambda(lambda tensor : torch.flip(tensor, dims=[0]))]),
        'shear': T.RandomAffine(10, shear=30)
    }
    skip_label_transformation_directory = {
         'gray_balance_adjust': T.ColorJitter(brightness=0.5, contrast=0.5),
    }
    volume_transforms = []
    skip_label_transforms = []
    for tf in transformations:
        if tf in volume_transformation_directory:
            volume_transforms.append(volume_transformation_directory[tf])
        if tf in skip_label_transformation_directory:
            skip_label_transforms.append(skip_label_transformation_directory[tf])
    is_label = False
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
        is_label = True
    tensor = torch.stack([tensor[0], tensor[0], tensor[0]])
    # z, c, x, y
    tensor = tensor.permute(3, 0, 1, 2)
    torch.manual_seed(seed)
    if training and aug:
        tensor = tensor.byte()
        #transformer = T.Compose(transforms)
        transformer = T.Compose(volume_transforms)
        tensor = transformer(tensor)
        if not is_label:
            skip_label_transformer = T.Compose(skip_label_transforms)
            tensor = skip_label_transformer(tensor)
        tensor = tensor.float()
    # Normalize by the average and standard deviation of the dataset
    if not is_label:
        normalize = T.Normalize(mean=avg, std=std)
        tensor = normalize(tensor)
    tensor = tensor.permute(1, 2, 3, 0)
    tensor = tensor[0].unsqueeze(0)
    if is_label:
        tensor = torch.squeeze(tensor, 0)
    return tensor

def get_dataloader(root, aug, batch_size, gpu, transforms, world_size=None, rank=None):
    train_set = Dataset(os.path.join(root, 'train'), True, aug, gpu, transforms)
    valid_set = Dataset(os.path.join(root, 'validation'), False, False, gpu, transforms)

    if world_size is None or rank is None:
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, drop_last=False)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=0, drop_last=False)
        return train_loader, valid_loader
    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, drop_last=True)
    valid_sampler = DistributedSampler(valid_set, num_replicas=world_size, rank=rank, drop_last=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, drop_last=False, sampler=train_sampler)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=0, drop_last=False, sampler=valid_sampler)

    return train_loader, valid_loader

def gen_args(args, defaults, file_args=None, config=None):
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
    if type(full_args['num_patches']) == int:
        full_args['num_patches'] = [full_args['num_patches'], full_args['num_patches']]
    elif len(full_args['num_patches']) == 1:
        full_args['num_patches'] = [full_args['num_patches'][0], full_args['num_patches'][0]]
    return SimpleNamespace(**full_args)
