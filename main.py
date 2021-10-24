import argparse
import json
import shutil
import os
from types import SimpleNamespace

import torch.multiprocessing as mp

from train import train_model
from optimize import optimize
from utils import gen_args
from conversions import gen_label_images, gen_targets
from models import model_mappings as rcnn_models
from evaluate import run_apply

from MatCNN.image2npy import convert_dir
from MatCNN.evaluate import run_stitch
from MatCNN.evaluate import run_apply as mat_run_apply
from MatCNN.evaluate import run_metrics as mat_run_metrics
from MatCNN.train import train as mat_train
from MatCNN.optimize import optimize as mat_optimize
from MatCNN.models import model_listing as mat_models

def main(args: SimpleNamespace):
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
    data_dir = os.path.join(args.root, args.name)
    if not os.path.isdir(args.run_dir):
        os.mkdir(args.run_dir)
    if not os.path.isdir(data_dir) and os.path.isfile(data_dir + '.tar'):
        shutil.unpack_archive(data_dir + '.tar', extract_dir = args.root)
    if 'gen_labels' in args.mode:
        gen_label_images(os.path.join(args.root, args.json_dir), os.path.join(args.root, 'labels'))
        valid_mode = True
    if 'image2npy' in modes:
        convert_dir(args)
        valid_mode = True
    if 'gen_targets' in args.mode:
        gen_targets(args)
        valid_mode = True
    if args.model in rcnn_models.keys():
        if 'train' in args.mode:
            if args.gpu and args.mp:
                mp.spawn(train_model, nprocs=args.gpu, args=(args,))
            train_model(args.gpu, args)
            valid_mode = True
        if 'apply' in modes:
            run_apply(args)
            valid_mode = True
        if 'metrics' in modes:
            #TODO: Add metrics functionality
            #run_metrics(args)
            valid_mode = True
        if 'optimize' in args.mode:
            optimize(args)
            valid_mode = True
    elif args.model in mat_models:
        if 'train' in modes:
            if args.gpu and args.mp:
                mp.spawn(mat_train, nprocs=args.gpu, args=(args,))
            else:
                mat_train(args.gpu, args)
            valid_mode = True
        if 'apply' in modes:
            mat_run_apply(args)
            valid_mode = True
        if 'stitch' in modes:
            run_stitch(args)
            valid_mode = True
        if 'metrics' in modes:
            #TODO: Add metrics functionality
            mat_run_metrics(args)
            valid_mode = True
        if 'optimize' in args.mode:
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='Root directory for the dataset')
    parser.add_argument('--lr', type=float, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train for', dest='epoch')
    parser.add_argument('--epoch', type=int, help='Number of epochs to train for')
    parser.add_argument('--gpu', type=int, help='Device ordinal for CUDA GPU')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--opt', type=str, help='Optimizer to train with. Either \'adam\' or \'sgd\'')
    parser.add_argument('--momentum', type=float, help='Momentum for SGD')
    parser.add_argument('--aug', help='Whether or not to augment inputs', action='store_true')
    parser.add_argument('--test_dir', type=str, help='Directory to test model on')
    parser.add_argument('--save', type=str, help='File name of saving destination')
    parser.add_argument('--mode', type=str, nargs='+', help='Mode to run. Available modes include \'train\'')
    parser.add_argument('--mp', action='store_true', help='Whether or not to use multiprocessing')
    parser.add_argument('--nodes', type=int, help='Number of nodes for multiprocessing')
    parser.add_argument('--nr', type=int, help='Index of the current node\'s first rank')
    parser.add_argument('--transforms', type=str, nargs='*', help='Which augmentations to use. Choose some combination of \'vertical_flip\', \'horizontal_flip\', and \'rotation\'')
    parser.add_argument('--model', type=str, help='Which model architecture to use. Available models include \'mask_rcnn\' and \'faster_rcnn\'')
    parser.add_argument('--config', type=str, help='Location of a full configuration file')
    parser.add_argument('--json_dir', type=str, help='Directory containing JSON files')
    parser.add_argument('--num_images', type=int, help='Number of images to train on')
    parser.add_argument('--checkpoint', type=str, help='Directory to a pth file to load')
    parser.add_argument('--no-prompt', dest='prompt', action='store_false', help='Skip prompt for checkpoint loading')
    parser.add_argument('--sampling_models', type=str, nargs='+', help='Models to sample from in HPO')
    parser.add_argument('--min_lr', type=float, help='Minimum learning rate for HPO')
    parser.add_argument('--max_lr', type=float, help='Maximum learning rate for HPO')
    parser.add_argument('--min_epochs', type=int, help='Minimum number of epochs for HPO')
    parser.add_argument('--max_epochs', type=int, help='Maximum number of epochs for HPO')
    parser.add_argument('--min_momentum', type=float, help='Minimum momentum for HPO')
    parser.add_argument('--max_momentum', type=float, help='Maximum momentum for HPO')
    parser.add_argument('--max_patch_size', type=int, help='Maximum patch size for HPO')
    parser.add_argument('--min_patch_size', type=int, help='Minimum patch size for HPO')
    parser.add_argument('--max_batch_size', type=int, help='Maximum batch size for HPO')
    parser.add_argument('--min_batch_size', type=int, help='Minimum batch size for HPO')
    parser.add_argument('--max_gamma', type=float, help='Max gamma value for HPO')
    parser.add_argument('--min_gamma', type=float, help='Min gamma value for HPO')
    parser.add_argument('--num_samples', type=int, help='Number of trials to run for HPO')
    parser.add_argument('--optimizers', type=str, nargs='+', help='Optimizers to sample from in HPO')
    parser.add_argument('--resume', action='store_true', help='Whether or not to resume an HPO trial')
    parser.add_argument('--jobs', type=int, help='Number of processes to run')
    parser.add_argument('--name', type=str, help='Name of the model')
    parser.add_argument('--graph', action='store_true', help='Whether or not to save a plot of the results')
    parser.add_argument('--run_dir', type=str, help='Directory to use for parsl')
    parser.add_argument('--loss', type=str, help='Loss function to use.')
    parser.add_argument('--version', type=str, help='Version of the network (for MatCNN compatibility)')
    parser.add_argument('--patience', type=int, help='Use patience after how many epochs')
    parser.add_argument('--image_size', type=int, nargs=2, help='Size of images processed as x y (necessary for evaluate re-stitching)')
    parser.add_argument('--num_patches', type=int, nargs='+', help='Number of patches each slice is split into. Set to 0 for automatic choice based on patch and image size.')
    parser.add_argument('--overlap_size', type=int, help='Width/height of the overlap between each patch')
    parser.add_argument('--slices', type=int, help='Number of slices to take at a time')
    parser.add_argument('--no-overlay', help='On evaluate, create full stitched images rather than overlays (if applicable)', dest='overlay', action='store_false')
    parser.add_argument('--data-split', type=float, nargs=3, help="3 floats for test validation train split of data in image2npy (in order test, validation, train)", dest='split')
    parser.add_argument('--collect', action='store_true', help='Whether or not to manually garbage collect each input')
    parser.add_argument('--patching_mode', type=str, help="Mode with which to patch images. Either \"grid\" or \"random\"")
    parser.add_argument('--clear_predictions', action='store_true', help='Clear prediction patches during evaluate')
    parser.add_argument('--tar', action='store_true', help='Tar output when done')
    parser.add_argument('--sampling_slices', type=int, nargs='+', help='Numbers of slices to use for models in HPO')
    parser.add_argument('--sampling_patching_modes', type=str, nargs='+', help='Patching modes to sample with in HPO')
    parser.add_argument('--min_overlap_size', type=int, help='Smallest size of overlap in HPO')
    parser.add_argument('--max_overlap_size', type=int, help='Maximum size of overlap in HPO')
    parser.add_argument('--mask', type=str, help='Mask to apply when getting metrics')
    parser.add_argument('--losses', type=str, nargs='+', help='Losses to sample from in HPO')
    parser.add_argument('--train_set', type=str, help='Dataset to use only as train')
    parser.add_argument('--test_set', type=str, help='Dataset to use only as test')
    parser.add_argument('--val_set', type=str, help='Dataset to use only as validation')
    parser.add_argument('--blocks', type=int, help='Number of encoding blocks to use in each architecture')
    parser.add_argument('--patch_size', type=int, help='Size of patches to use for target generation')
    parser.add_argument('--data_split', type=float, nargs=3, help="3 floats for test validation train split of data in image2npy (in order test, validation, train)", dest='split')
    parser.add_argument('--gamma', type=float, help='Amount to decrease LR by every 3 epochs')

    args = parser.parse_args()
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
        'min_lr': 0.0000001,
        'max_lr': 0.1,
        'min_epochs': 5,
        'max_epochs': 100,
        'min_momentum': 0.7,
        'max_momentum': 0.95,
        'max_patch_size': 600,
        'min_patch_size': 200,
        'max_batch_size': 2,
        'min_batch_size': 1,
        'max_gamma': 0.01,
        'min_gamma': 0.0000001,
        'num_samples': 100,
        'sampling_models': ['mask_rcnn', 'faster_rcnn', 'retina_net'],
        'optimizers': ['sgd', 'adam'],
        'resume': False,
        'jobs': 4,
        'name': 'bubbles',
        'graph': False,
        'run_dir': 'runinfo',
        'loss': 'cross_entropy',
        'aug': False,
        'version': None,
        'patience': 0,
        'image_size': [852, 852],
        'num_patches': [3],
        'overlap_size': None,
        'slices': 8,
        'overlay': True,
        'split': [0.1, 0.2, 0.7],
        'patching_mode': 'grid',
        'collect': False,
        'clear_predictions': False,
        'tar': True,
        'sampling_slices': [4, 6, 8],
        'sampling_patching_modes': ['grid'],
        'min_overlap_size': 10,
        'max_overlap_size': 30,
        'losses': ['cross_entropy'],
        'mask': None,
        'train_set': None,
        'test_set': None,
        'val_set': None,
        'blocks': 2,
        'patch_size': 300,
        'split': [0.1, 0.2, 0.7],
        'gamma': 0.001,
    }
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
        full_args = gen_args(args, full_args, file_args=json_args, config=config)
        full_args.world_size = full_args.gpu * full_args.nodes
        main(full_args)
