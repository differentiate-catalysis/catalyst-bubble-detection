import argparse
import json
import os
from types import SimpleNamespace

import torch.multiprocessing as mp

from evaluate import apply, test
from train import train_model
from utils import gen_args


def main(args: SimpleNamespace):
    if 'train' in args.mode:
        if args.gpu and args.mp:
            mp.spawn(train_model, nprocs=args.gpu, args=(args,))
        train_model(args.gpu, args)
    if 'test' in args.mode:
        test(args)
    if 'apply' in args.mode:
        apply(args)

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
    parser.add_argument('--mode', type=str, nargs='+', help='Mode to run. Either \'train\' or \'test\'')
    parser.add_argument('--mp', action='store_true', help='Whether or not to use multiprocessing')
    parser.add_argument('--nodes', type=int, help='Number of nodes for multiprocessing')
    parser.add_argument('--nr', type=int, help='Index of the current node\'s first rank')
    parser.add_argument('--transforms', type=str, nargs='*', help='Which augmentations to use. Choose some combination of \'vertical_flip\', \'horizontal_flip\', and \'rotation\'')
    parser.add_argument('--model', type=str, help='Which model architecture to use')
    parser.add_argument('--config', type=str, help='Location of a full configuration file')
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
        'config': None
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
