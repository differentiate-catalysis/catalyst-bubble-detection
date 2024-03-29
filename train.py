import math
import os
from types import SimpleNamespace
from typing import List

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import ray
import torch
import torch.distributed as dist
import torch.utils.data
from ray import tune
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, Optimizer
from torch.utils.data import DataLoader

from evaluate import evaluate
from models import model_mappings
from utils import get_dataloaders, tqdm, warmup_lr_scheduler


def train_model(gpu: int, args: SimpleNamespace):
    '''
    General training function for models. Handles initializing necessary objects,
    tracking metrics, and calling train and evaluation loops.
    Parameters
    ----------
    gpu: int
        CUDA ordinal for what GPU to run on
    args: SimpleNamespace
        Namespace of all options/hyperparameters
    '''
    loss_train_list = []
    loss_val_list = []
    if args.mp:
        world_size = args.world_size
        if not tune.is_session_enabled():
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
    train_loader, valid_loader = get_dataloaders(args, world_size, rank)
    model = model_mappings(args).to(device)
    if args.mp:
        model = DDP(model, device_ids=[gpu])
    # Only touch trainable parameters
    params = [p for p in model.parameters() if p.requires_grad]
    if args.opt.lower() == 'adam':
        optimizer = Adam(params, lr=args.lr)
    elif args.opt.lower() == 'sgd':
        optimizer = SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=0.0005)
    else:
        raise ValueError('Invalid optimizer specified')
    # Lower LR every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=args.gamma)
    iou_max = 0
    model_dir = os.path.join(args.save, '%s.pth' % args.name)
    epoch_init = 1
    last_loss_train = float('inf')
    last_loss_eval = float('inf')
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        try:
            if args.mp:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            if not tune.is_session_enabled():
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Successfuly loaded checkpoint')
        except:
            print('Failed to load checkpoint. Training from scratch...')
    elif os.path.isfile(model_dir):
        continue_training = input('Found a model that was already saved! Load from checkpoint? (y/n) ').lower() if args.prompt else 'n'
        if continue_training == 'y' or continue_training == 'ye' or continue_training == 'yes':
            try:
                checkpoint = torch.load(model_dir)
                if args.mp:
                    model.module.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                last_loss_train = checkpoint['loss_train']
                last_loss_eval = checkpoint['loss_val']
                epoch_init = checkpoint['epoch'] + 1
                iou_max = checkpoint['iou_val_max']
                print('Successfuly loaded checkpoint, continuing from epoch %d of %d...' % (epoch_init, args.epoch))
            except:
                print('Failed to load checkpoint. Training from scratch...')
        else:
            print('Did not load checkpoint. Training from scratch...')
    for epoch in range(epoch_init, args.epoch + 1):
        # Train for one epoch
        loss_train = train_epoch(model, optimizer, train_loader, epoch, amp=amp, gpu=gpu)
        lr_scheduler.step()

        # Evaluate
        loss_eval, iou_eval, mAP_eval = evaluate(model, valid_loader, amp=amp, gpu=gpu)

        # Append losses to list to graph later
        loss_train_list.append(loss_train)
        loss_val_list.append(loss_eval)

        # Sync results across all ranks
        if args.mp:
            results = torch.zeros((args.world_size, 4), device=device, dtype=torch.float32)
            results[:] = torch.tensor([loss_train, loss_eval, iou_eval, mAP_eval], device=device, dtype=torch.float32)
            # Sync the results from all nodes
            for i in range(args.world_size):
                amp = amp
                dist.broadcast(results[i], src=i)
            # Combine the results and send to the rest of the group
            results = torch.mean(results, dim=0)
            loss_train, loss_eval, iou_eval, mAP_eval = results.to(torch.device('cpu'))

        # Stop on overfit
        if loss_train - last_loss_train < 0 and loss_eval - last_loss_eval > 0:
            print('Possible overfitting, stopping early.')
            break

        # Update last losses
        last_loss_train = loss_train
        last_loss_eval = last_loss_eval

        if tune.is_session_enabled():
            if isinstance(loss_eval, torch.Tensor):
                loss_eval = loss_eval.to(torch.device('cpu')).item()
            if isinstance(iou_eval, torch.Tensor):
                iou_eval = iou_eval.to(torch.device('cpu')).item()
            if isinstance(mAP_eval, torch.Tensor):
                mAP_eval = mAP_eval.to(torch.device('cpu')).item()

            tune.report(loss=loss_eval, iou=iou_eval, map=mAP_eval)
            with args.ray_checkpoint_dir(step=epoch) as checkpoint_dir:
                model_dir = os.path.join(checkpoint_dir, os.pardir, 'last.pth')

        # Checkpoint if good result, only checkpoint on one rank
        if (iou_eval > iou_max or tune.is_session_enabled()) and (rank == None or rank == 0):
            iou_max = iou_eval
            if not os.path.isdir(args.save):
                os.mkdir(args.save)
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
            }, model_dir)

        if args.graph:
            if not os.path.isdir(os.path.join(args.save, 'plots')):
                os.mkdir(os.path.join(args.save, 'plots'))
            graph(loss_train_list, loss_val_list, os.path.join(args.save, 'plots', '%s.png' %(args.name)))


def train_epoch(model: Module, optimizer: Optimizer, train_loader: DataLoader, epoch: int, amp: bool, gpu: int) -> float:
    '''
    Train loop for models.
    Parameters
    ----------
    model: Module
        A torchvision compatible object detector, such as Faster-RCNN or Mask-RCNN.
    optimizer: Optimizer
        A PyTorch optimizer with the model parameters.
    train_loader: DataLoader
        Dataloader for the training data
    epoch: int
        Current training epoch
    amp: bool
        Whether or not to use Accelerated Mixed Precision - lowers memory requirements
    gpu: int
        CUDA ordinal for which GPU to run on
    '''
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
            print('Loss is %s, stopping training' % loss_value)
            return 10000

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        if lr_scheduler is not None:
            lr_scheduler.step()
        optimizer.zero_grad()

    print('--- training result ---')
    print('loss: %.5f' % (total_loss / num_samples))

    return total_loss / num_samples

def graph(loss_train_list: List, loss_val_list: List, save_path: str):
    '''
    Graphs the train and validation loss, writes it to a file.
    Parameters
    ----------
    loss_train_list: List
        List containing the loss at each epoch during training
    loss_val_list: List
        List containing the loss during validation at the end of each epoch
    save_path: str
        Path to the file to write the graph to
    '''
    fig, ax = plt.subplots()
    x = range(len(loss_train_list))
    ax.plot(x, loss_train_list, label='Training Loss')
    ax.plot(x, loss_val_list, label='Validation Loss')
    ax.set_xlim(0, len(loss_train_list) - 1)
    plt.xticks(range(len(loss_train_list)))
    ax.set(xlabel='Epoch', ylabel='Loss',
           title='Loss of Model')
    plt.legend(loc="upper left")
    plt.savefig(save_path)
    print('Plot saved')
