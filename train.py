import math
import os
import sys
from types import SimpleNamespace
from typing import Tuple, Union

import ray
import torch
import torch.distributed as dist
import torch.utils.data
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate import evaluate
from models import model_mappings
from utils import get_dataloaders, warmup_lr_scheduler
import matplotlib.pyplot as plt


def train_model(gpu: int, args: SimpleNamespace):
    loss_train_list = []
    loss_val_list = []
    if args.mp:
        world_size = args.world_size
        if not hasattr(args, 'ray_checkpoint_dir'):
            rank = args.nr * args.gpu + gpu
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=args.world_size,
                rank=rank
            )
        else:
            rank = args.nr * args.gpu + int(ray.get_gpu_ids()[0])
    else:
        world_size = None
        rank = None
    torch.manual_seed(0)
    if gpu != -1 and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        amp = args.amp
        device = torch.device('cuda', gpu)
    else:
        gpu = -1
        amp = False
        device = torch.device('cpu')
    model = model_mappings[args.model].to(device)
    if args.mp:
        model = DDP(model, device_ids=[gpu])
    # Only touch trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    if args.opt.lower() == 'adam':
        optimizer = Adam(params, lr=args.lr)
    elif args.opt.lower() == 'sgd':
        optimizer = SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    else:
        raise ValueError('Invalid optimizer specified')
    # Lower LR every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    train_loader, valid_loader = get_dataloaders(args, world_size, rank)
    iou_max = 0
    last_loss_train = float('inf')
    last_loss_eval = float('inf')
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        loss_train = train_epoch(model, optimizer, train_loader, epoch, amp=amp, gpu=gpu)
        lr_scheduler.step()

        # Evaluate
        loss_eval, iou_eval = evaluate(model, valid_loader, amp=amp, gpu=gpu)

        # Append losses to list to graph later
        loss_train_list.append(loss_train)
        loss_val_list.append(loss_eval)

        # Sync results across all ranks
        if args.mp:
            results = torch.zeros((args.world_size, 3), device=device, dtype=torch.float32)
            results[:] = torch.tensor([loss_train, loss_eval, iou_eval], device=device, dtype=torch.float32)
            # Sync the results from all nodes
            for i in range(args.world_size):
                amp = amp
                dist.broadcast(results[i], src=i)
            # Combine the results and send to the rest of the group
            results = torch.mean(results, dim=0)
            loss_train, loss_eval, iou_eval = results.to(torch.device('cpu'))

        # Stop on overfit
        if loss_train - last_loss_train < 0 and loss_eval - last_loss_eval > 0:
            print('Possible overfitting, stopping early.')
            break

        # Update last losses
        last_loss_train = loss_train
        last_loss_eval = last_loss_eval

        # Checkpoint if good result, only checkpoint on one rank
        if iou_eval > iou_max and (rank == None or rank == 0):
            iou_max = iou_eval
            if not os.path.isdir('saved'):
                os.mkdir('saved')
            if args.mp:
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model_state_dict,
                'loss_val': loss_eval,
                'loss_train': loss_train,
                'iou_val_max': iou_max
            }, 'saved/best.pth')

        graph(loss_train_list, loss_val_list, 'saved/loss.png')


def train_epoch(model: Module, optimizer: Optimizer, train_loader: DataLoader, epoch: int, amp: bool, gpu: int) -> Tuple[float, float]:
    if gpu != -1:
        device = torch.device('cuda', gpu)
    else:
        device = torch.device('cpu')
    model.train()
    num_samples = train_loader.batch_size * len(train_loader)

    lr_scheduler = None
    if epoch == 1:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(train_loader) - 1)
        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    for images, targets in tqdm(train_loader, desc='Epoch %d' % (epoch)):
        images = list(image.to(device) for image in images)
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=amp):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            total_loss += loss_value * train_loader.batch_size

        if not math.isfinite(loss_value):
            print(targets)
            print('Loss is %s, stopping training' % loss_value)
            print(loss_dict)
            sys.exit(1)

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        if lr_scheduler is not None:
            lr_scheduler.step()
        optimizer.zero_grad()

    print('--- training result ---')
    print('loss: %.5f' % (total_loss / num_samples))

    return total_loss / num_samples

def graph(loss_train_list, loss_val_list, save_path):
    fig, ax = plt.subplots()
    x = range(len(loss_train_list))
    ax.plot(x, loss_train_list, label='Training Loss')
    ax.plot(x, loss_val_list, label='Validation Loss')
    ax.set(xlabel='Epoch', ylabel='Loss',
           title='Loss of Model')
    plt.savefig(save_path)
    print('Plot saved')