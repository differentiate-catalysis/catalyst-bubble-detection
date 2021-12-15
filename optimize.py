import json
import math
import os
import random
from functools import partial
from types import SimpleNamespace
from typing import Dict

import numpy.random as nprd
import ray
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.torch import DistributedTrainableCreator, distributed_checkpoint_dir
from ray.tune.schedulers import ASHAScheduler

from conversions import gen_targets
from models import model_keys
from train import train_model
from transforms import transform_mappings
from utils import gen_args


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
        gen_targets(args)
    train_model(0, args)


def pick_model(args: SimpleNamespace):
    if not set(args.sampling_models).issubset(set(model_keys)):
        raise ValueError('Sampling models contain invalid values')
    def f(spec):
        model = random.choice(args.sampling_models)
        return model.lower()
    return f


def pick_optimizer(args: SimpleNamespace):
    if not set(args.optimizers).issubset(set(['adam', 'adamw', 'sgd'])):
        raise ValueError('Optimizers to sample from contain invalid values')
    def f(spec):
        return random.choice(args.optimizers)
    return f


def pick_transforms(args: SimpleNamespace):
    if not set(args.transforms).issubset(set(transform_mappings)):
        raise ValueError('Transforms to sample from contain invalid values')
    def f(spec):
        return random.sample(args.transforms, nprd.randint(0, len(args.transforms) + 1))
    return f


def pick_batch_size(args: SimpleNamespace):
    def f(spec):
        max_exp = int(math.log(args.max_batch_size, 2))
        min_exp = int(math.log(args.min_batch_size, 2))
        exp = nprd.randint(min_exp, max_exp + 1)
        return 2 ** exp
    return f


def pick_patch_size(args: SimpleNamespace):
    if args.min_patch_size > args.max_patch_size:
        raise ValueError('Min patch size is larger than max patch size')
    def f(spec):
        min_valid_64 = args.min_patch_size // 64 + 1
        max_valid_64 = args.max_patch_size // 64
        return nprd.randint(min_valid_64, max_valid_64 + 1) * 64
    return f


def optimize(args):
    if len(args.root) > 0 and args.root[0] != '/':
        args.root = os.path.join(os.getcwd(), args.root)
    name_dir = os.path.join('saved', args.name)
    hyperparam_config = {
        'model': tune.sample_from(pick_model(args)),
        'opt': tune.sample_from(pick_optimizer(args)),
        'lr': tune.loguniform(args.min_lr, args.max_lr),
        'momentum': tune.uniform(args.min_momentum, args.max_momentum),
        'epoch': tune.sample_from(lambda _: nprd.randint(args.min_epochs, args.max_epochs+1)),
        'patch_size': tune.sample_from(pick_patch_size(args)),
        'batch_size': tune.sample_from(pick_batch_size(args)),
        'transforms': tune.sample_from(pick_transforms(args)),
        'gamma': tune.loguniform(args.min_gamma, args.max_gamma),
    }
    if not os.path.isdir(name_dir):
        os.mkdir(name_dir)
    scheduler = ASHAScheduler(
            metric='map',
            mode='max',
            max_t=args.max_epochs,
            grace_period=1,
            reduction_factor=2
    )
    reporter = CLIReporter(metric_columns=['loss', 'map', 'iou'])
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
    best_trial = result.get_best_trial('map', 'max', 'last')
    df = result.dataframe(metric='map', mode='max')
    df.to_csv(os.path.join(name_dir, 'trials.csv'))
    if best_trial is not None:
        print('Best trial config: {}'.format(best_trial.config))
        print('Best trial final validation loss: {}'.format(
            best_trial.last_result['loss']))
        print('Best trial final validation mAP: {}'.format(
            best_trial.last_result['map']))
        print('Best trial final validation iou: {}'.format(
            best_trial.last_result['iou']))
        print('Best Checkpoint Dir: ' + str(best_trial.checkpoint.value))
        checkpoint = [file for file in os.listdir(best_trial.checkpoint.value) if 'best.pth' in file]
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
        config_out['root'] = args.root#'data/HandSeg_Reserved'
        config_out['mp'] = args.mp
        config_out['gpu'] = args.gpu
        config_out['amp'] = args.amp
        config_out['patch'] = args.patch_size
        config_out['name'] = '%s_optimized' % args.name
        config_out['version'] = 0
        config_out['test_dir'] = 'test'
        with open(os.path.join(name_dir, 'best.json'), 'w') as f:
            json.dump(config_out, f, indent=4)
