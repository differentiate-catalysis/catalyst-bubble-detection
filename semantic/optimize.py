from functools import partial
import os
from typing import Dict

import torch
import numpy.random as nprd

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from types import SimpleNamespace
from semantic.train import train
from semantic.evaluate import run_apply
from semantic.image2npy import convert_dir
from semantic.utils import gen_args
import random
import math
from ray.tune.integration.torch import DistributedTrainableCreator, distributed_checkpoint_dir
import json

def optim_train(config: Dict, checkpoint_dir: str = None, defaults: Dict = None, args_space: SimpleNamespace = None, dir: str = None):
    if dir:
        os.chdir(dir)
    if defaults is None:
        defaults = vars(args_space).copy()
        defaults['num_patches'] = 0
        defaults['prompt'] = False
        defaults['tar'] = True
        defaults['model'] = 'unet'
    args = gen_args(args=SimpleNamespace(**config), defaults=defaults)
    args.jobs = args.jobs // args.gpu
    if args.patch_size == 852:
        args.num_patches = 1
        args.overlap_size = 0
    if args.mp:
        args.ray_checkpoint_dir = distributed_checkpoint_dir
    else:
        args.ray_checkpoint_dir = tune.checkpoint_dir
    if checkpoint_dir is not None:
        args.checkpoint = checkpoint_dir
    gpu = int(ray.get_gpu_ids()[0])
    if not args.mp:
        args.name += '_gpu_%d' % gpu
    if (args.mp and gpu == 0) or not args.mp:
        convert_dir(args)
    train(0, args)
    # if (args.mp and gpu == 0) or not args.mp:
        # args.root = args.test_root
        # args.mp = False
        # args.gpu = 0
        # run_apply(args)

# Assuming 40 GB VRAM and AMP
def get_max_patch_size(model: str, slices: int):
    if model == 'unet':
        if slices == 8:
            max_patch = 852
        elif slices == 6:
            max_patch = 852
        elif slices == 4:
            max_patch = 852
    elif model == 'fcdensenet':
        if slices == 8:
            max_patch = 852
        if slices == 6:
            max_patch = 852
        if slices == 4:
            max_patch = 852
    elif model == 'unet2d':
        max_patch = 852
    else:
        raise ValueError('Model and slices combination is invalid')
    return max_patch
# Assuming 12 GB VRAM
# def get_max_patch_size(model: str, slices: int):
    # if model == 'unet':
        # if slices == 12:
            # max_patch = 450
        # elif slices == 8:
            # max_patch = 500
        # elif slices == 6:
            # max_patch = 650
        # elif slices == 4:
            # max_patch = 852
    # elif model == 'fcdensenet':
        # if slices == 12:
            # max_patch = 350
        # elif slices == 8:
            # max_patch = 400
    # elif model == 'unet2d':
        # max_patch = 852
    # else:
        # raise ValueError('Model and slices combination is invalid')
    # return max_patch

def pick_model(args: SimpleNamespace):
    if not set(args.sampling_models).issubset(set(['unet', 'fcdensenet', 'unet2d'])):
        raise ValueError('Sampling models contain invalid values')
    def f(spec):
        model = random.choice(args.sampling_models)
        return model.lower()
    return f

def pick_loss(args:SimpleNamespace):
    if not set(args.losses).issubset(set(['abl', 'cross_entropy', 'lovasz', 'ce+iou', 'ce+iabl'])):
        raise ValueError('Sampling losses contain invalid values')
    def f(spec):
        loss = random.choice(args.losses)
        return loss
    return f

def pick_slices(args: SimpleNamespace):
    if not set(args.sampling_slices).issubset(set([1, 4, 6, 8])):
        raise ValueError('Slices to sample from contains invalid values')
    def f(spec):
        model = spec.config.model
        if model == 'unet':
            return random.choice(list(set(args.sampling_slices) & set([4, 6, 8])))
        elif model == 'fcdensenet':
            return random.choice(list(set(args.sampling_slices) & set([4, 6, 8])))
        elif model == 'unet2d':
            return 1
    return f

def pick_patch_size(args: SimpleNamespace):
    min_patch_arg = args.min_patch_size
    max_patch_arg = args.max_patch_size
    def f(spec):
        model = spec.config.model
        slices = spec.config.slices
        #max_patch = get_max_patch_size(model, slices)
        #max_patch = min(max_patch_arg, max_patch)
        patch_size = 64 * nprd.randint(min_patch_arg // 64, max_patch_arg // 64 + 1)
        return patch_size
    return f

def pick_batch_size(spec):
    model = spec.config.model
    slices = spec.config.slices
    patch_size = spec.config.patch_size
    max_patch = get_max_patch_size(model, slices)
    batch_size = int(math.log(max_patch / patch_size, 2))
    batch_size = 2 ** nprd.randint(0, batch_size + 1)
    return batch_size

def pick_overlap_size(args: SimpleNamespace):
    def f(spec):
        min_size = args.min_overlap_size
        max_size = args.max_overlap_size
        return nprd.randint(min_size, max_size + 1)
    return f

def pick_optimizer(args: SimpleNamespace):
    if not set(args.optimizers).issubset(set(['adam', 'adamw', 'sgd'])):
        raise ValueError('Optimizers to sample from contain invalid values')
    def f(spec):
        return random.choice(args.optimizers)
    return f

def pick_transforms(args: SimpleNamespace):
    if not set(args.transforms).issubset(set(['rotation', 'shear', 'horizontal_flip', 'vertical_flip', 'gray_balance_adjust'])):
        raise ValueError('Transforms to sample from contain invalid values')
    def f(spec):
        return random.sample(args.transforms, nprd.randint(1, len(args.transforms) + 1))
    return f

def pick_patching_mode(args: SimpleNamespace):
    if not set(args.sampling_patching_modes).issubset(set(['grid', 'random'])):
        raise ValueError('Sampling modes contain invalid values')
    def f(spec):
        return random.choice(args.sampling_patching_modes)
    return f

def optimize(args):
    if len(args.root) > 0 and args.root[0] != '/':
        args.root = os.path.join(os.getcwd(), args.root)
    name_dir = os.path.join('saved', args.name)
    hyperparam_config = {
        'model': tune.sample_from(pick_model(args)),
        'slices': tune.sample_from(pick_slices(args)),
        'patch_size': tune.sample_from(pick_patch_size(args)),
        'batch_size': tune.sample_from(pick_batch_size),
        'overlap_size': tune.sample_from(pick_overlap_size(args)),
        'opt': tune.sample_from(pick_optimizer(args)),
        'lr': tune.loguniform(args.min_lr, args.max_lr),
        'epoch': tune.sample_from(lambda _: nprd.randint(args.min_epochs, args.max_epochs+1)),
        'aug': tune.choice([True, False]),
        'patience': tune.sample_from(lambda spec: nprd.randint(1, spec.config.epoch+2)),
        'transforms': tune.sample_from(pick_transforms(args)),
        'patching_mode': tune.sample_from(pick_patching_mode(args)),
        'loss': tune.sample_from(pick_loss(args)),
    }
    if not os.path.isdir(name_dir):
        os.mkdir(name_dir)
    scheduler = ASHAScheduler(
            metric='iou', #convert to BF1
            mode='max',
            max_t=args.max_epochs,
            grace_period=1,
            reduction_factor=2
    )
    reporter = CLIReporter(metric_columns=['loss', 'accuracy', 'iou'])
    source_dir = os.getcwd()
    if args.mp:
        trainer = DistributedTrainableCreator(
            partial(optim_train, args_space=args, dir=source_dir),
            backend='nccl',
            num_gpus_per_worker=1,
            num_workers=args.gpu,
            num_cpus_per_worker=1
        )
        trial_resources = None
    else:
        trainer = partial(optim_train, args_space=args, dir=source_dir)
        trial_resources = {'cpu': args.jobs//args.gpu, 'gpu': 1 if args.gpu > -1 else 0}
    result = tune.run(
            trainer,
            name = args.name,
            resources_per_trial = trial_resources,
            config = hyperparam_config,
            local_dir = 'saved/tune',
            num_samples = args.num_samples,
            scheduler = scheduler,
            progress_reporter = reporter,
            resume=args.resume,
    )
    best_trial = result.get_best_trial('iou', 'max', 'last')
    if best_trial is not None:
        print('Best trial config: {}'.format(best_trial.config))
        print('Best trial final validation loss: {}'.format(
            best_trial.last_result['loss']))
        print('Best trial final validation accuracy: {}'.format(
            best_trial.last_result['accuracy']))
        print('Best trial final validation iou: {}'.format(
            best_trial.last_result['iou']))
        print('Best Checkpoint Dir: ' + str(best_trial.checkpoint.value))
        checkpoint = [file for file in os.listdir(best_trial.checkpoint.value) if 'pth' in file and args.name in file]
        if len(checkpoint) > 0:
            checkpoint = checkpoint[0]
        else:
            raise RuntimeError('No valid checkpoint found. Trial likely did not complete properly')

        checkpoint = torch.load(os.path.join(best_trial.checkpoint.value, checkpoint))
        torch.save({
                    'epoch': best_trial.config['epoch'],
                    'version': best_trial.config['epoch'],
                    'optimizer_state_dict': checkpoint['optimizer_state_dict'],
                    'model_state_dict': checkpoint['model_state_dict'],
                }, os.path.join(name_dir, 'best.pth')
        )
        config_out = best_trial.config.copy()
        config_out['num_patches'] = [-(args.image_size[0] // -config_out['patch_size']), -(args.image_size[1] // -config_out['patch_size'])]
        config_out['root'] = 'data/HandSeg_Reserved'
        config_out['mp'] = args.mp
        config_out['gpu'] = args.gpu
        config_out['amp'] = args.amp
        config_out['name'] = '%s_optimized' % args.name
        config_out['version'] = 0
        config_out['test_dir'] = 'test'
        config_out['image_size'] = args.image_size
        config_out['blocks'] = args.blocks
        with open(os.path.join(name_dir, 'best.json'), 'w') as f:
            json.dump(config_out, f, indent=4)
