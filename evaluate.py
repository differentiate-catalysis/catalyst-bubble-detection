import os
from types import SimpleNamespace

import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from utils import Dataset, collate_fn


def evaluate(model: Module, valid_loader: DataLoader, amp: bool, gpu: int):
    if gpu != -1:
        device = torch.device('cuda', gpu)
    else:
        device = torch.device('cpu')

    with torch.no_grad():
        # Set model to evaluate mode (disable batchnorm and dropout, output boxes)
        model.eval()
        total_iou = 0
        total_loss = 0
        num_samples = valid_loader.batch_size * len(valid_loader)
        for images, targets in tqdm(valid_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(images)
                for i in range(len(targets)):
                    output = outputs[i]
                    target = targets[i]
                    iou = torchvision.ops.box_iou(output['boxes'], target['boxes'])
                    dims = iou.shape
                    matrix = torch.zeros((max(dims[0], dims[1]), max(dims[0], dims[1])))
                    matrix[:dims[0], :dims[1]] = iou
                    total_iou += torch.sum(torch.diagonal(matrix))
                # Must set model to train mode to get loss
                model.train()
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values()).item()
                model.eval()
            total_loss += loss * valid_loader.batch_size
        print('--- evaluation result ---')
        print('loss: %.5f, iou %.5f' % (total_loss / num_samples, total_iou / num_samples))

    return total_loss / num_samples, total_iou / num_samples
