import argparse
import json
import os
from types import SimpleNamespace

import torch.multiprocessing as mp

from train import train_model
from utils import gen_args

from conversions import gen_label_images

def main(args: SimpleNamespace):
    if 'gen_labels' in args.mode:
        gen_label_images(os.path.join(args.root, args.json_dir), os.path.join(args.root, 'labels'))
    elif 'train' in args.mode:
        if args.gpu and args.mp:
            mp.spawn(train_model, nprocs=args.gpu, args=(args,))
        for num_images in args.dataset_sizes:
            args.save = 'save_' + str(num_images)
            args.num_images = num_images
            train_model(args.gpu, args)
    else:
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
    parser.add_argument('--dataset_sizes', type=list, help='Sizes of dataset to train')
    args = parser.parse_args()
    defaults = {
        'root': 'data/bubbles',
        'gpu': -1,
        'lr': 0.0001,
        'batch_size': 1,
        'epochs': 10,
        'amp': False,
        'opt': 'adam',
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
        'dataset_sizes': [1, 3, 5, 15, 28]
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
