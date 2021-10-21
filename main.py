import argparse
import json
import os
from types import SimpleNamespace

import torch.multiprocessing as mp

from train import train_model
from optimize import optimize
from utils import gen_args

from conversions import gen_label_images, gen_targets

def main(args: SimpleNamespace):
    valid_mode = False
    if not os.path.isdir(args.run_dir):
        os.mkdir(args.run_dir)
    if 'gen_labels' in args.mode:
        gen_label_images(os.path.join(args.root, args.json_dir), os.path.join(args.root, 'labels'))
        valid_mode = True
    if 'gen_targets' in args.mode:
        gen_targets(args)
        valid_mode = True
    if 'train' in args.mode:
        if args.gpu and args.mp:
            mp.spawn(train_model, nprocs=args.gpu, args=(args,))
        train_model(args.gpu, args)
        valid_mode = True
    if 'optimize' in args.mode:
        optimize(args)
        valid_mode = True
    if not valid_mode:
        raise ValueError('No valid mode supplied.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='Root directory for the dataset')
    parser.add_argument('--lr', type=float, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs to train for')
    parser.add_argument('--gpu', type=int, help='Device ordinal for CUDA GPU')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')
    parser.add_argument('--opt', type=str, help='Optimizer to train with. Either \'adam\' or \'sgd\'')
    parser.add_argument('--momentum', type=float, help='Momentum for SGD')
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
    parser.add_argument('--num_samples', type=int, help='Number of trials to run for HPO')
    parser.add_argument('--optimizers', type=str, nargs='+', help='Optimizers to sample from in HPO')
    parser.add_argument('--resume', action='store_true', help='Whether or not to resume an HPO trial')
    parser.add_argument('--jobs', type=int, help='Number of processes to run')
    parser.add_argument('--name', type=str, help='Name of the model')
    parser.add_argument('--graph', action='store_true', help='Whether or not to save a plot of the results')
    parser.add_argument('--run_dir', type=str, help='Directory to use for parsl')
    parser.add_argument('--patch_size', type=int, help='Size of patches to use for target generation')
    parser.add_argument('--data_split', type=float, nargs=3, help="3 floats for test validation train split of data in image2npy (in order test, validation, train)", dest='split')
    parser.add_argument('--max_patch_size', type=int, help='Maximum patch size for HPO')
    parser.add_argument('--min_patch_size', type=int, help='Minimum patch size for HPO')

    args = parser.parse_args()
    defaults = {
        'root': 'data/bubbles',
        'gpu': -1,
        'lr': 0.0001,
        'batch_size': 1,
        'epochs': 10,
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
        'num_samples': 100,
        'sampling_models': ['mask_rcnn', 'faster_rcnn', 'retina_net'],
        'optimizers': ['sgd', 'adam'],
        'resume': False,
        'jobs': 4,
        'name': 'bubbles',
        'graph': False,
        'run_dir': 'runinfo',
        'patch_size': 300,
        'split': [0.1, 0.2, 0.7],
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
