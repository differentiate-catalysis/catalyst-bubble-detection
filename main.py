import argparse
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import List

import torch.multiprocessing as mp

from arg_checks import check_args
from conversions import apply_transforms, gen_label_images, gen_targets
from evaluate import run_apply, run_metrics
from MatCNN.evaluate import run_apply as mat_run_apply
from MatCNN.evaluate import run_metrics as mat_run_metrics
from MatCNN.evaluate import run_stitch
from MatCNN.image2npy import convert_dir
from MatCNN.models import model_listing as mat_models
from MatCNN.optimize import optimize as mat_optimize
from MatCNN.train import train as mat_train
from models import model_keys as rcnn_models
from optimize import optimize
from train import train_model
from utils import gen_args
from arg_checks import check_args, condense_args



def main(args: SimpleNamespace, changed: List[str], explicit: List[str]):
    if not os.path.isdir('saved'):
        os.mkdir('saved')
    if not os.path.isdir('saved/log'):
        os.mkdir('saved/log')
    if not os.path.isdir('saved/plots'):
        os.mkdir('saved/plots')
    if type(args.mode) is str:
        args.mode = [args.mode]
    modes  = [mode.lower() for mode in args.mode]
    valid_mode = False
    check_args(modes, changed, explicit)
    args = condense_args(args, modes)
    data_dir = os.path.join(args.root, args.name)
    if not os.path.isdir(args.run_dir):
        os.mkdir(args.run_dir)
    if not os.path.isdir(data_dir) and os.path.isfile(data_dir + '.tar'):
        shutil.unpack_archive(data_dir + '.tar', extract_dir = args.root)
    if 'gen_labels' in modes:
        gen_label_images(os.path.join(args.root, args.json_dir), os.path.join(args.root, 'labels'))
        valid_mode = True
    if 'augment' in modes:
        apply_transforms(args)
        valid_mode = True
    if 'image2npy' in modes:
        convert_dir(args)
        valid_mode = True
    if 'gen_targets' in modes:
        gen_targets(args)
        valid_mode = True
    first_model = args.model[0] if type(args.model) is list else args.model
    if first_model in rcnn_models:
        if 'train' in modes:
            if args.gpu and args.mp:
                mp.spawn(train_model, nprocs=args.gpu, args=(args,))
            train_model(args.gpu, args)
            valid_mode = True
        if 'evaluate' in modes:
            loss = run_apply(args)
            run_metrics(args, loss=loss)
            valid_mode = True
        if 'apply' in modes:
            if args.video_dir is not None:
                videos = []
                video_formats = ['mp4', 'mov']
                for format in video_formats:
                    videos.extend(list(Path(args.video_dir).rglob('*.%s' % format.lower())))
                    videos.extend(list(Path(args.video_dir).rglob('*.%s' % format.upper())))
                videos = set(videos)
                print('Found %d videos.' % len(videos))
                root = args.root
                for video in videos:
                    video = str(video)
                    args.video = video
                    basename = os.path.basename(args.video)
                    dot_index = basename.rfind('.')
                    if dot_index != -1:
                        basename = basename[: dot_index]
                    args.root = os.path.join(root, basename)
                    run_apply(args)
            else:
                run_apply(args)
            valid_mode = True
        if 'metrics' in modes:
            run_metrics(args)
            valid_mode = True
        if 'optimize' in modes:
            optimize(args)
            valid_mode = True
    elif first_model in mat_models:
        if 'train' in modes:
            if args.gpu and args.mp:
                mp.spawn(mat_train, nprocs=args.gpu, args=(args,))
            else:
                mat_train(args.gpu, args)
            valid_mode = True
        if 'evaluate' in modes:
            loss = mat_run_apply(args)
            run_stitch(args)
            mat_run_metrics(args, loss=loss)
            valid_mode = True
        if 'apply' in modes:
            mat_run_apply(args)
            valid_mode = True
        if 'stitch' in modes:
            run_stitch(args)
            valid_mode = True
        if 'metrics' in modes:
            mat_run_metrics(args)
            valid_mode = True
        if 'optimize' in modes:
            mat_optimize(args)
            valid_mode = True
    if not valid_mode:
        raise ValueError('No valid mode supplied.')
    if args.tar:
        if 'optimize' in modes:
            if args.gpu > -1:
                for gpu in range(args.gpu):
                    shutil.rmtree(os.path.join(args.root, args.name + '_gpu_%d' % gpu))
            else:
                shutil.rmtree(data_dir)
        else:
            shutil.make_archive(data_dir, 'tar', args.root, args.name)
            shutil.rmtree(data_dir)


if __name__ == '__main__':

    defaults = {
        'root': 'data/bubbles',
        'gpu': -1,
        'lr': 0.0001,
        'batch_size': 1,
        'epoch': 10,
        'amp': False,
        'opt': 'adam',
        'momentum': 0.9,
        'test_dir': 'test',
        'save': 'saved',
        'mode': ['train'],
        'mp': False,
        'nodes': 1,
        'nr': 0,
        'transforms': ['rotation', 'horizontal_flip', 'vertical_flip'],
        'model': 'faster_rcnn',
        'config': None,
        'json_dir': 'json',
        'num_images': -1,
        'prompt': True,
        'checkpoint': None,
        'num_samples': 100,
        'resume_hpo': False,
        'jobs': 4,
        'name': 'bubbles',
        'graph': False,
        'run_dir': 'runinfo',
        'loss': 'cross_entropy',
        'aug': False,
        'version': None,
        'patience': 0,
        'image_size': [1920, 1080],
        'num_patches': [3],
        'overlap_size': None,
        'slices': 8,
        'overlay': True,
        'split': [0.1, 0.2, 0.7],
        'patching_mode': 'grid',
        'collect': False,
        'clear_predictions': False,
        'tar': False,
        'mask': None,
        'train_set': None,
        'test_set': None,
        'val_set': None,
        'blocks': 2,
        'patch_size': 300,
        'split': [0.1, 0.2, 0.7],
        'gamma': 0.001,
        'imagenet_stats': False,
        'stats_file': None,
        'video': None,
        'video_dir': None,
        'simclr_checkpoint': None,
        'augment_out': 'data/augment',
        'data_workers': 0,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='Root directory for the dataset. Default: %s' % defaults['root'])
    parser.add_argument('--lr', type=float, nargs='*', help='Learning rate for the optimizer. Default: %s' % defaults['lr'])
    parser.add_argument('--batch_size', type=int, nargs='*', help='Batch size. Default: %s' % defaults['batch_size'])
    parser.add_argument('--epoch', type=int, nargs='*', help='Number of epochs to train for. Default: %s' % defaults['epoch'])
    parser.add_argument('--gpu', type=int, help='Device ordinal for CUDA GPU. Set to -1 to run on CPU only. Default: %s' % defaults['gpu'])
    parser.add_argument('--amp', action='store_true', help='Use mixed precision (reduces VRAM requirements). Default: %s' % defaults['amp'])
    parser.add_argument('--opt', type=str, nargs='*', help='Optimizer to train with. Either \'adam\' or \'sgd\'. Default: %s' % defaults['opt'])
    parser.add_argument('--momentum', type=float, nargs='*', help='Momentum for SGD. Default: %s' % defaults['momentum'])
    parser.add_argument('--aug', help='Whether or not to augment inputs', action='store_true') #TODO: remove
    parser.add_argument('--test_dir', type=str, help='Directory to test model on. Default: %s' % defaults['test_dir'])
    parser.add_argument('--save', type=str, help='Directory to write saved model weights to. Subdirectories will be created for HPO runs. Default: %s' % defaults['save'])
    parser.add_argument('--mode', type=str, nargs='+', help='Mode to run. Available modes include \'train\', \'apply\', \'evaluate\', and \'optimize\'')
    parser.add_argument('--mp', action='store_true', help='Whether or not to use multiprocessing. Only use if you are training with multiple GPUs or multiple nodes. Default: %s' % defaults['mp'])
    parser.add_argument('--nodes', type=int, help='Number of nodes for multiprocessing. Only necessary if mp is set.')
    parser.add_argument('--nr', type=int, help='Index of the current node\'s first rank. Only necessary if mp is set.')
    parser.add_argument('--transforms', type=str, nargs='*', help='Which augmentations to use. Choose some combination of \'vertical_flip\', \'horizontal_flip\', and \'rotation\'. Default: %s' % defaults['transforms'])
    parser.add_argument('--model', type=str, nargs='*', help='Which model architecture to use. Available models include \'mask_rcnn\', \'faster_rcnn\', \'retina_net\', and \'simclr_faster_rcnn\'. Default: %s' % defaults['model'])
    parser.add_argument('--config', type=str, help='Location of a full configuration file. This contains options for a given run.')
    parser.add_argument('--json_dir', type=str, help='Directory containing several JSON configuration files.')
    parser.add_argument('--num_images', type=int, help='Number of images to train on') # idt this is used tbh
    parser.add_argument('--checkpoint', type=str, help='Filepath to a set of saved weights (pth file) to load')
    parser.add_argument('--no-prompt', dest='prompt', action='store_false', help='Skip prompt for checkpoint loading. Default: %s' % (not defaults['prompt']))
    parser.add_argument('--num_samples', type=int, help='Number of trials to run for HPO')
    parser.add_argument('--resume_hpo', action='store_true', help='Whether or not to resume an HPO trial')
    parser.add_argument('--jobs', type=int, help='Number of cores to use for parsl')
    parser.add_argument('--data_workers', type=int, help='Number of processes to run for dataloading')
    parser.add_argument('--name', type=str, help='Name of the model')
    parser.add_argument('--graph', action='store_true', help='Whether or not to save a plot of the results')
    parser.add_argument('--run_dir', type=str, help='Directory to use for parsl')
    parser.add_argument('--loss', type=str, nargs='*', help='Loss function to use.')
    parser.add_argument('--version', type=str, help='Version of the network (for MatCNN compatibility)')
    parser.add_argument('--patience', type=int, help='Use patience after how many epochs')
    parser.add_argument('--image_size', type=int, nargs=2, help='Size of images processed as x y (necessary for evaluate re-stitching)')
    parser.add_argument('--num_patches', type=int, nargs='+', help='Number of patches each slice is split into. Set to 0 for automatic choice based on patch and image size.')
    parser.add_argument('--overlap_size', type=int, nargs='*', help='Width/height of the overlap between each patch')
    parser.add_argument('--slices', type=int, nargs='*', help='Number of slices to take at a time')
    parser.add_argument('--no-overlay', help='On evaluate, create full stitched images rather than overlays (if applicable)', dest='overlay', action='store_false')
    parser.add_argument('--data-split', type=float, nargs=3, help="3 floats for test validation train split of data in image2npy (in order test, validation, train)", dest='split')
    parser.add_argument('--collect', action='store_true', help='Whether or not to manually garbage collect each input')
    parser.add_argument('--patching_mode', type=str, nargs='*', help="Mode with which to patch images. Either \"grid\" or \"random\"")
    parser.add_argument('--clear_predictions', action='store_true', help='Clear prediction patches during evaluate')
    parser.add_argument('--tar', action='store_true', help='Tar output when done')
    parser.add_argument('--mask', type=str, help='Mask to apply when getting metrics')
    parser.add_argument('--train_set', type=str, help='Dataset to use only as train')
    parser.add_argument('--test_set', type=str, help='Dataset to use only as test')
    parser.add_argument('--val_set', type=str, help='Dataset to use only as validation')
    parser.add_argument('--blocks', type=int, help='Number of encoding blocks to use in each architecture')
    parser.add_argument('--patch_size', type=int, nargs='*', help='Size of patches to use for target generation. Set to -1 to disable patching')
    parser.add_argument('--data_split', type=float, nargs=3, help="3 floats for test validation train split of data in image2npy (in order test, validation, train)", dest='split')
    parser.add_argument('--gamma', type=float, nargs='*', help='Amount to decrease LR by every 3 epochs')
    parser.add_argument('--imagenet_stats', action='store_true', help='Use ImageNet stats instead of dataset stats for normalization')
    parser.add_argument('--stats_file', type=str, help='JSON file to read stats from')
    parser.add_argument('--video', type=str, help='Video file to perform inference on')
    parser.add_argument('--video_dir', type=str, help='Directory containing videos. Recursively searches for videos')
    parser.add_argument('--simclr_checkpoint', type=str, help='Weights for SimCLR pre-trained ResNet')
    parser.add_argument('--augment_out', type=str, help='Output directory for augment mode')

    args = parser.parse_args()
    if args.config:
        if os.path.isfile(args.config):
            configs = [args.config]
        elif os.path.isdir(args.config):
            configs = [os.path.join(args.config, file) for file in os.listdir(args.config) if file.endswith('.json')]
        else:
            configs = [None]
    else:
        configs = [None]
    for config in configs:
        full_args = defaults.copy()
        json_args = None
        if config: #Start by subbing in from config file
            with open(config, 'r') as fp:
                json_args = json.load(fp)
        full_args, changed_args, explicit_args = gen_args(args, full_args, file_args=json_args, config=config)
        full_args.world_size = full_args.gpu * full_args.nodes
        main(full_args, changed_args, explicit_args)
