import os
import time
import gc
from types import SimpleNamespace
from typing import Tuple

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.modules.module import Module
import torch.optim as optim
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from MatCNN.metrics import accuracy
from MatCNN.utils import AverageMeter, Recorder, get_dataloader
from MatCNN.models import get_model
from MatCNN.losses import loss_mappings
from MatCNN.evaluate import evaluate
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from ray import tune
import ray

def train(gpu: int, args: SimpleNamespace):
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
    if torch.cuda.is_available() and args.gpu > -1:
        torch.cuda.set_device(gpu)
        criterion = loss_mappings[args.loss].cuda(gpu)
    else:
        if args.gpu > -1:
            print('No CUDA capable GPU detected. Using CPU...')
        else:
            print('Configuration has GPU set to -1. Using CPU...')
        criterion = loss_mappings[args.loss]
    if torch.cuda.is_available() and args.gpu > -1:
        device = torch.device('cuda', gpu)
        model = get_model(args.model, args.blocks)
        model = model.to(device)
        if args.mp:
            model = DDP(model, device_ids=[gpu])
    else:
        device = torch.device('cpu')
        model = get_model(args.model, args.blocks)
        model = model.to(device)
    start = time.time()
    if args.opt.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.opt.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.opt.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    else:
        raise ValueError("Supplied optimizer is invalid. Optimizer must be adam, sgd, or adamw")
    train_loader, valid_loader = get_dataloader(os.path.join(args.root, args.name), args.aug, args.batch_size, gpu, args.transforms, world_size=world_size, rank=rank)
    scheduler = ReduceLROnPlateau(optimizer, patience=args.patience)
    recorder = Recorder(('loss_train', 'acc_train', 'loss_val', 'acc_val', 'mean_iou', 'class_precision', 'class_iou'))
    iou_val_max = 0
    model_dir = 'saved/%s.pth' % (args.name)
    log_dir = 'saved/log/%s.log' % (args.name)
    plot_dir = 'saved/plots/%s.png' % (args.name)
    epoch_init = 1
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        try:
            if args.mp:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            if not hasattr(args, 'ray_checkpoint_dir'):
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if os.path.isfile(log_dir):
                recorder.load(torch.load(log_dir))
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
                if os.path.isfile(log_dir):
                    recorder.load(torch.load(log_dir))
                loss_train = checkpoint['loss_train']
                loss_val = checkpoint['loss_val']
                epoch_init = checkpoint['epoch'] + 1
                iou_val_max = checkpoint['iou_val_max']
                print('Successfuly loaded checkpoint, continuing from epoch %d of %d...' % (epoch_init, args.epoch))
            except:
                print('Failed to load checkpoint. Training from scratch...')
        else:
            print('Did not load checkpoint. Training from scratch...')
    last_loss_train = float('inf')
    last_loss_val = float('inf')
    for epoch in range(epoch_init, args.epoch + 1):
        loss_train, acc_train = train_epoch(model, criterion, optimizer, train_loader, epoch, use_amp=args.amp, gpu=gpu, collect=args.collect)
        loss_val, acc_val, iou_val, class_precision, class_iou = evaluate(model, criterion, valid_loader, gpu=gpu, collect=args.collect, use_amp=args.amp)
        if args.mp:
            # Communicate results between nodes
            results = torch.zeros((args.world_size, 7), device=device, dtype=torch.float32)
            results[:] = torch.tensor([loss_train, acc_train, loss_val, acc_val, iou_val, class_precision, class_iou], device=device, dtype=torch.float32)
            # Sync the results from all nodes
            for i in range(args.world_size):
                dist.broadcast(results[i], src=i)
            # Combine the results and send to the rest of the group
            results = torch.mean(results, dim=0)
            loss_train, acc_train, loss_val, acc_val, iou_val, class_precision, class_iou = results.to(torch.device('cpu'))

        if loss_train - last_loss_train < 0 and loss_val - last_loss_val > 0:
            print("Possible overfitting, stopping early.")
            break
        last_loss_train = loss_train
        last_loss_val = loss_val
        recorder.update((loss_train, acc_train, loss_val, acc_val, iou_val, class_precision, class_iou))

        scheduler.step(loss_train)

        if hasattr(args, 'ray_checkpoint_dir'):
            if rank == None or rank == 0:
                with args.ray_checkpoint_dir(step=epoch) as checkpoint_dir:
                    model_dir = os.path.join(checkpoint_dir, '%s.pth' % args.name)
                    log_dir = os.path.join(checkpoint_dir, '%s.log' % args.name)
                if args.mp:
                    model_state_dict = model.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                torch.save(recorder.record, log_dir)
                torch.save({
                    'epoch': epoch,
                    'version': args.version,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': model_state_dict,
                    'loss_val': loss_val,
                    'loss_train': loss_train,
                    'iou_val_max': iou_val
                }, model_dir)

            if args.mp:
                loss_report = loss_val.item()
                iou_report = iou_val.item()
                acc_report = acc_val.item()
            else:
                loss_report = loss_val
                iou_report = iou_val
                acc_report = acc_val

            tune.report(loss=loss_report, iou=iou_report, accuracy=acc_report)
            with args.ray_checkpoint_dir(step=epoch) as checkpoint_dir:
                model_dir = os.path.join(checkpoint_dir, '../best.pth')
                log_dir = os.path.join(checkpoint_dir, '../best.log')

        if (iou_val > iou_val_max or hasattr(args, 'ray_checkpoint_dir')) and (rank == None or rank == 0):
            if args.mp:
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            torch.save(recorder.record, log_dir)
            torch.save({
                'epoch': epoch,
                'version': args.version,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model_state_dict,
                'loss_val': loss_val,
                'loss_train': loss_train,
                'iou_val_max': iou_val
            }, model_dir)
            print('validation iou improved from %.5f to %.5f. Model Saved.' % (iou_val_max, iou_val))
            iou_val_max = iou_val

        if (optimizer.param_groups[0]['lr'] / args.lr) <= 1e-3:
            print('Learning Rate Reduced to 1e-3 of Original Value', 'Training Stopped', sep='\n')
            break
    end = time.time()
    time_taken = end - start
    print(recorder.record)

def train_epoch(model: Module, criterion: _Loss, optimizer: Optimizer, trainloader: DataLoader, epoch: int, use_amp: bool = True, gpu: int = 0, collect: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Accuracy', ':6.4f')
    model.train()
    if gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', gpu)
    scaler = GradScaler(enabled=(gpu != -1 and use_amp))
    for t, (inputs, labels, _) in enumerate(tqdm(trainloader, desc='Epoch ' + str(epoch))):
        inputs, labels = inputs.to(device), labels.to(device).long()
        with torch.cuda.amp.autocast(enabled=(gpu != -1 and use_amp)):
            outputs = model(inputs)
            predictions = outputs.argmax(1)
            loss = criterion(outputs, labels)

        accs.update(accuracy(predictions, labels), inputs.shape[0])
        losses.update(loss.item(), inputs.size(0))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if collect:
            del inputs, labels, outputs
            gc.collect()
            if torch.cuda.is_available() and gpu != -1:
                torch.cuda.empty_cache()
    print('--- training result ---')
    print('loss: %.5f, accuracy: %.5f' % (losses.avg, accs.avg))
    return losses.avg, accs.avg
