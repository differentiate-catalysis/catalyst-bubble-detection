import argparse
import os
import shutil
import json
from types import SimpleNamespace

from train import train
from evaluate import run_apply, run_stitch, run_metrics
from optimize import optimize
from image2npy import convert_dir
from utils import gen_args

import torch.multiprocessing as mp

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
    if 'test' in modes:
        print("Ran " + args.config)
        valid_mode = True
    if 'image2npy' in modes:
        convert_dir(args)
        valid_mode = True
    if 'train' in modes:
        if args.gpu and args.mp:
            mp.spawn(train, nprocs=args.gpu, args=(args,))
        else:
            train(args.gpu, args)
        valid_mode = True
    if 'evaluate' in modes:
        loss = run_apply(args)
        run_stitch(args)
        run_metrics(args, loss=loss)
        valid_mode = True
    if 'apply' in modes:
        run_apply(args)
        valid_mode = True
    if 'stitch' in modes:
        run_stitch(args)
        valid_mode = True
    if 'metrics' in modes:
        run_metrics(args)
        valid_mode = True
    if 'optimize' in modes:
        optimize(args)
        valid_mode = True
    if not valid_mode:
        print('No valid mode supplied. Run again with \"--mode [test, image2npy, train, evaluate, apply, stitch, metrics, optimize]\"')
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

def fill_args(args):
    defaults = {
        'root': 'data/Exp116',
        'loss': 'cross_entropy',
        'gpu': 1,
        'lr': 0.005,
        'batch_size': 2,
        'epoch': 10,
        'aug': False,
        'name': None,
        'version': None,
        'mode': 'train',
        'opt': 'adam',
        'amp': False,
        'patience': 0,
        'test_dir': 'test',
        'transforms': ['rotation', 'horizontal_flip', 'vertical_flip'],
        'image_size': [852, 852],
        'num_patches': [3],
        'patch_size': None,
        'overlap_size': None,
        'slices': 8,
        'config': None,
        'overlay': True,
        'jobs': 1,
        'prompt': True,
        'checkpoint': None,
        'split': [0.1, 0.2, 0.7],
        'patching_mode': 'grid',
        'skip_evaluate': False,
        'nodes': 1,
        'nr': 0,
        'mp': False,
        'collect': False,
        'clear_predictions': False,
        'tar': False,
        'min_lr': 0.000001,
        'max_lr': 0.001,
        'min_epochs': 5,
        'max_epochs': 15,
        'num_samples': 1,
        'resume': True,
        'skip_metrics': False,
        'mask': 'tomography',
        'sampling_models': ['unet', 'fcdensenet'],
        'sampling_slices': [4, 6, 8],
        'min_patch_size': 80,
        'max_patch_size': 852,
        'min_overlap_size': 10,
        'max_overlap_size': 30,
        'optimizers': ['adam', 'adamw', 'sgd'],
        'sampling_patching_modes': ['grid'],
        'losses': ['cross_entropy'],
        'train_set': None,
        'test_set': None,
        'val_set': None,
        'run_dir': 'runinfo',
        'blocks': 2
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Model architecture to run. Either \"unet\" or \"fcdensenet\"')
    parser.add_argument('--loss', type=str, help='Loss function to use.')
    parser.add_argument('--root', type=str, help='Root directory for the experiment data')
    parser.add_argument('--gpu', type=int, help='CUDA Device Ordinal to use, or number of GPUs per node if multiprocessing. Set to -1 to run on CPU.')
    parser.add_argument('--lr', type=float, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epoch', type=int, help='Epochs to train for')
    parser.add_argument('--aug', help='Whether or not to augment input volumes', action='store_true')
    parser.add_argument('--name', type=str, help='Name of the network')
    parser.add_argument('--version', type=str, help='Version of the network')
    parser.add_argument('--mode', type=str, nargs='+', help='Mode to run: test, image2npy, train, evaluate, or optimize')
    parser.add_argument('--opt', type=str, help="adam or sgd")
    parser.add_argument('--amp', help='Whether or not to use mixed precision', action='store_true')
    parser.add_argument('--patience', type=int, help='Use patience after how many epochs')
    parser.add_argument('--test_dir', type=str, help='Directory with test data')
    parser.add_argument('--transforms', type=str, help='Transforms to apply to the data (only used if aug is true)', nargs='+')
    parser.add_argument('--image_size', type=int, nargs=2, help='Size of images processed as x y (necessary for evaluate re-stitching)')
    parser.add_argument('--num_patches', type=int, nargs='+', help='Number of patches each slice is split into. Set to 0 for automatic choice based on patch and image size.')
    parser.add_argument('--patch_size', type=int, help='Width/height of each patch.')
    parser.add_argument('--overlap_size', type=int, help='Width/height of the overlap between each patch')
    parser.add_argument('--slices', type=int, help='Number of slices to take at a time')
    parser.add_argument('--config', type=str, help='Location of a full configuration file')
    parser.add_argument('--no-overlay', help='On evaluate, create full stitched images rather than overlays (if applicable)', dest='overlay', action='store_false')
    parser.add_argument('--jobs', help='Number of jobs to run concurrently (only relevant in image2npy mode)', type=int)
    parser.add_argument('--no-prompt', help='Do not prompt for directory wiping or checkpoint loading (will default to yes for directory wiping, no for checkpoint loading)', dest='prompt', action='store_false')
    parser.add_argument('--checkpoint', type=str, help="Checkpoint (or previously trained model) to resume training from")
    parser.add_argument('--data-split', type=float, nargs=3, help="3 floats for test validation train split of data in image2npy (in order test, validation, train)", dest='split')
    parser.add_argument('--patching_mode', type=str, help="Mode with which to patch images. Either \"grid\" or \"random\"")
    parser.add_argument('--skip_evaluate', action='store_true', help="Use to skip network evaluation in evaluate (and skip straight to overlay/stitching)")
    parser.add_argument('--nodes', default=1, type=int, metavar='N', help="Number of nodes to run on")
    parser.add_argument('--nr', default=0, type=int, help='Ranking within the nodes')
    parser.add_argument('--mp', action='store_true', help='Whether or not to use multiprocessing')
    parser.add_argument('--collect', action='store_true', help='Whether or not to manually garbage collect each input')
    parser.add_argument('--clear_predictions', action='store_true', help='Clear prediction patches during evaluate')
    parser.add_argument('--tar', action='store_true', help='Tar output when done')
    parser.add_argument('--min_lr', type=float, help='Minimum learning rate for the optimizer during HPO')
    parser.add_argument('--max_lr', type=float, help='Maximum learning rate for the optimizer during HPO')
    parser.add_argument('--min_epochs', type=int, help='Minimum epochs during HPO')
    parser.add_argument('--max_epochs', type=int, help='Maximum epochs during HPO')
    parser.add_argument('--sampling_models', type=str, nargs='+', help='Which models to sample from in HPO')
    parser.add_argument('--sampling_slices', type=int, nargs='+', help='Numbers of slices to use for models in HPO')
    parser.add_argument('--sampling_patching_modes', type=str, nargs='+', help='Patching modes to sample with in HPO')
    parser.add_argument('--min_patch_size', type=int, help='Smallest patch size to sample in HPO')
    parser.add_argument('--max_patch_size', type=int, help='Maximum patch size to sample in HPO')
    parser.add_argument('--min_overlap_size', type=int, help='Smallest size of overlap in HPO')
    parser.add_argument('--max_overlap_size', type=int, help='Maximum size of overlap in HPO')
    parser.add_argument('--optimizers', type=str, nargs='+', help='Which optimizers to sample from in HPO')
    parser.add_argument('--num_samples', type=int, help='Number of trials during HPO')
    parser.add_argument('--resume', help='Whether or not to resume an HPO run', action='store_true')
    parser.add_argument('--skip_metrics', action='store_true', help='Skip metric generation in evaluate calls')
    parser.add_argument('--mask', type=str, help='Mask to apply when getting metrics')
    parser.add_argument('--losses', type=str, nargs='+', help='Losses to sample from in HPO')
    parser.add_argument('--train_set', type=str, help='Dataset to use only as train')
    parser.add_argument('--test_set', type=str, help='Dataset to use only as test')
    parser.add_argument('--val_set', type=str, help='Dataset to use only as validation')
    parser.add_argument('--run_dir', type=str, help='Directory to use for parsl')
    parser.add_argument('--blocks', type=int, help='Number of encoding blocks to use in each architecture')
    
    args = parser.parse_args()