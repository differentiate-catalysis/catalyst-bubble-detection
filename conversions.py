import json
import math
import os
import random
import shutil

import numpy as np
import parsl
import torch
from parsl import python_app
from parsl.config import Config
from parsl.executors.threads import ThreadPoolExecutor
from PIL import Image, ImageDraw
from typing import Dict, List, Optional, Tuple
import torchvision.transforms.functional as F

import cv2

from transforms import target_from_masks
from utils import get_circle_coords, tqdm, get_transforms


def v7labstobasicjson(infile: str, outfile: str=None) -> dict:
    """
    Converts v7labs style json to basic json file: UNUSED

    Args:
        infile (str): v7 labs json file
        outfile (str, optional): file to output basic json to. Defaults to None.

    Returns:
        dict: returned directory of relevant values
    """
    with open(infile, 'r') as infp:
        indata = json.load(infp)
        outdir = {}
        outdir['width'] = indata['image']['width']
        outdir['height'] = indata['image']['height']
        outdir['filename'] = indata['image']['filename']
        outdir['bubbles'] = []
        for annotate in indata['annotations']:
            bubble = annotate['ellipse']
            bubble_data = {
                "center": [bubble["center"]["x"], bubble["center"]["y"]],
                "radius": bubble["radius"]["x"]
            }
            outdir['bubbles'].append(bubble_data)
    if outfile:
        with open(outfile, 'w') as outfp:
            json.dump(outdir, outfp, indent=2)
    return outdir


def getjsontype(json_dict: dict) -> bool:
    """Returns True if the json_dict is in basic form, False if not (probably in v7 labs form and should convert): UNUSED

    Args:
        json_dict (dict): JSON dictionary of keys

    Returns:
        bool: whether json_dict is in basic form
    """
    #
    return not ({'bubbles', 'height', 'width', 'filename'} - set(json_dict.keys()))


def gen_label_images(indir: str, outdir: str) -> None:
    """For all files in indir, generates semantic segmentation label images for each json file

    Args:
        indir (str): Directory containing json files_dict
        outdir (str): Directory to store outputted image (.png) files
    """
    in_files = [os.path.join(indir, f) for f in os.listdir(indir) if f.endswith('.json')]
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    print("Generating labels")
    for file in tqdm(in_files):
        with open(file, 'r') as fp:
            json_data = json.load(fp)
        if not getjsontype(json_data):
            json_data = v7labstobasicjson(file)
        image = Image.new('1', (json_data['width'], json_data['height']))
        drawer = ImageDraw.Draw(image)
        for bubble in json_data['bubbles']:
            drawer.ellipse([bubble['center'][0] - bubble['radius'], bubble['center'][1] - bubble['radius'], bubble['center'][0] + bubble['radius'], bubble['center'][1] + bubble['radius']], fill = 1)
        image.save(os.path.join(outdir, os.path.splitext(os.path.split(json_data['filename'])[1])[0] + '.png'))


@python_app
def convert_to_targets(image_root, json_root, trial_dir, image_name, patch_size, splits, rand_num=None):
    """Convert v7 labs json file to .npy files for training.
    Also converts an image file into patches and generates training masks only containing bubbles in the specific patch.

    Args:
        image_root (str): Directory containg image
        json_root (str): Directory containing label json
        trial_dir (str): Directory to store generated labels and patches in
        image_name (str): Image file to process
        patch_size (int): Width and height of patches to generate
        splits ([float, float, float]): [test, validation, train] splits of data. Should add up to 1.
        rand_num (float, optional): Pre-generated random number to determine split to store image in. If None, random number is generated at this function's runtime.
    """
    image = Image.open(os.path.join(image_root, image_name)).convert(mode='RGB')
    width = image.width
    height = image.height
    has_label = os.path.isfile(os.path.join(json_root, image_name[:-4] + '.json'))
    if has_label:
        with open(os.path.join(json_root, image_name[:-4] + '.json')) as fp:
            json_data = json.load(fp)
            annotations = json_data['annotations']
            bubbles = []
            for circle in annotations:
                if 'ellipse' in circle:
                    ellipse = circle['ellipse']
                    center_x = ellipse['center']['x']
                    center_y = ellipse['center']['y']
                    radius_x = ellipse['radius']['x']
                    radius_y = ellipse['radius']['y']
                    bubbles.append(np.array([center_x, center_y, radius_x, radius_y]))

        # Allocate space for mask
        mask = np.zeros((len(bubbles), height, width), dtype=np.uint8)

        for i, bubble in enumerate(bubbles):
            center_x = bubble[0]
            center_y = bubble[1]
            radius_x = bubble[2]
            radius_y = bubble[3]
            # Generate points for mask
            rr, cc = get_circle_coords(center_y, center_x, radius_y, radius_x, height, width)
            mask[i, rr, cc] = 1

    # Prepare for patching
    image = np.asarray(image)
    if patch_size == -1:
        n_patches_w = 1
        n_patches_h = 1
    else:
        n_patches_w = math.ceil(width / patch_size)
        overlap_w = n_patches_w * patch_size - width
        n_overlaps_w = n_patches_w - 1
        # Concept: Be on the lookout for magic camels.
        overlaps_w = np.ones(n_patches_w, dtype=np.int32) * (overlap_w + (-overlap_w % n_overlaps_w)) // n_overlaps_w
        dec_indices = np.random.default_rng().choice(n_overlaps_w, size=((-overlap_w % n_overlaps_w)), replace=False) + 1
        overlaps_w[dec_indices] -= 1
        overlaps_w[0] = 0
        overlaps_w = np.cumsum(overlaps_w)

        n_patches_h = math.ceil(height / patch_size)
        overlap_h = n_patches_h * patch_size - height
        n_overlaps_h = n_patches_h - 1
        overlaps_h = np.ones(n_patches_h, dtype=np.int32) * (overlap_h + (-overlap_h % n_overlaps_h)) // n_overlaps_h
        dec_indices = np.random.default_rng().choice(n_overlaps_h, size=((-overlap_h % n_overlaps_h)), replace=False) + 1
        overlaps_h[dec_indices] -= 1
        overlaps_h[0] = 0
        overlaps_h = np.cumsum(overlaps_h)

    for i in range(n_patches_w):
        for j in range(n_patches_h):
            # Disable patching
            if patch_size == -1:
                patch = image
                if has_label:
                    masks = mask
            else:
                patch = image[patch_size * j - overlaps_h[j] : patch_size * (j + 1) - overlaps_h[j], patch_size * i - overlaps_w[i] : patch_size * (i + 1) - overlaps_w[i], :]
                if has_label:
                    masks = mask[:, patch_size * j - overlaps_h[j] : patch_size * (j + 1) - overlaps_h[j], patch_size * i - overlaps_w[i] : patch_size * (i + 1) - overlaps_w[i]]

            if has_label:
                target = target_from_masks(torch.from_numpy(masks))
                masks = target['masks'].numpy()
                boxes = target['boxes'].numpy()
                areas = target['areas'].numpy()

            image_out = Image.fromarray(patch, mode='RGB')


            if not has_label:
                target_output_dir = os.path.join(trial_dir, 'test', 'targets') # Must be test if no label
                image_output_dir = os.path.join(trial_dir, 'test', 'patches')
                image_out.save(os.path.join(image_output_dir, '%s_%d_%d.png' % (image_name[:-4], i, j)))
                np.save(os.path.join(target_output_dir, '%s_%d_%d_masks.npy' % (image_name[:-4], i, j)), np.array(0))
                np.save(os.path.join(target_output_dir, '%s_%d_%d_boxes.npy' % (image_name[:-4], i, j)), np.array(0))
                np.save(os.path.join(target_output_dir, '%s_%d_%d_areas.npy' % (image_name[:-4], i, j)), np.array(0))
            # Only save patches with bubbles in them
            elif 0 not in masks.shape and 0 not in boxes.shape:
                if rand_num is None:
                    rand_num = torch.rand(1)
                if rand_num < splits[0]:
                    split = 'test'
                elif rand_num < splits[0] + splits[1]:
                    split = 'validation'
                else:
                    split = 'train'
                split_dir = os.path.join(trial_dir, split)
                target_output_dir = os.path.join(split_dir, 'targets')
                image_output_dir = os.path.join(split_dir, 'patches')
                image_out.save(os.path.join(image_output_dir, '%s_%d_%d.png' % (image_name[:-4], i, j)))
                np.save(os.path.join(target_output_dir, '%s_%d_%d_masks.npy' % (image_name[:-4], i, j)), masks)
                np.save(os.path.join(target_output_dir, '%s_%d_%d_boxes.npy' % (image_name[:-4], i, j)), boxes)
                np.save(os.path.join(target_output_dir, '%s_%d_%d_areas.npy' % (image_name[:-4], i, j)), areas)


def gen_targets(args):
    """Generates label numpy files, patches images if necessary, and places in train, test, and validation directories.

    Args:
        args (SimpleNamespace): Namespace of patching, storage, and splits information
    """
    local_threads = Config(
        executors=[
            ThreadPoolExecutor(
                max_threads=args.jobs,
                label='local_threads'
            )
        ]
    )
    local_threads.checkpoint_mode = 'manual'
    run_dir = os.path.join(args.run_dir, args.name)
    if not os.path.isdir(run_dir):
        os.mkdir(run_dir)
    local_threads.run_dir = run_dir
    parsl.load(local_threads)
    workers = []
    splits = ['train', 'validation', 'test']
    trial_dir = os.path.join(args.root, args.name)
    if not os.path.isdir(trial_dir):
        os.mkdir(trial_dir)
    image_root = os.path.join(args.root, 'images')
    json_root = os.path.join(args.root, 'json')
    torch.manual_seed(8995)

    for split in splits:
        split_dir = os.path.join(trial_dir, split)
        if os.path.isdir(split_dir):
            delete = 'y'
            if args.prompt:
                delete = input('%s dir found! Wipe? (Y/n)' % split.capitalize())
            if delete.lower() in ['y', 'ye', 'yes']:
                shutil.rmtree(split_dir)
                os.mkdir(split_dir)
            else:
                continue
        else:
            os.mkdir(split_dir)
        target_output_dir = os.path.join(split_dir, 'targets')
        image_output_dir = os.path.join(split_dir, 'patches')
        if not os.path.isdir(target_output_dir):
            os.mkdir(target_output_dir)
        if not os.path.isdir(image_output_dir):
            os.mkdir(image_output_dir)
    if not (args.train_set or args.val_set or args.test_set):
        random.seed(8995)
        num_images = len(os.listdir(image_root))
        split_numbers = [i / num_images for i in range(num_images)]
        random.shuffle(split_numbers)
        for image_name in os.listdir(image_root):
            workers.append(convert_to_targets(image_root, json_root, trial_dir, image_name, args.patch_size, args.split, rand_num=split_numbers.pop()))
    else: #No specified sets, should randomly assign using given splits
        for t, split_dir in enumerate([args.test_set, args.val_set, args.train_set]):
            if split_dir:
                image_root = os.path.join(split_dir, 'images')
                json_root = os.path.join(split_dir, 'json')
                for image_name in os.listdir(image_root):
                    splits_sub = [0, 0, 0]
                    splits_sub[t] = 1
                    workers.append(convert_to_targets(image_root, json_root, trial_dir, image_name, args.patch_size, splits_sub, 0.1))
    for worker in tqdm(workers):
        result = worker.result()
    parsl.clear()

class ToTensorGPU():
    def init(self, gpu: int):
        self.gpu = gpu
    def forward(self, image: torch.Tensor, target: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        image = F.to_tensor(image).to(device=f'cuda:{self.gpu}')
        return image, target

def apply_transforms(args):
    transform_compose = get_transforms(True, args.transforms)
    if (args.gpu!=-1):
        transform_compose.transforms[0] = ToTensorGPU(args.gpu)
    images = [os.path.join(args.root, f) for f in os.listdir(args.root) if os.path.isfile(os.path.join(args.root, f))]
    for image_file in tqdm(images):
        image = Image.open(image_file).convert(mode='RGB')
        image, _ = transform_compose(image)
        image = image.numpy()
        image = np.transpose(image, (1, 2, 0))
        if not os.path.isdir(args.augment_out):
            os.makedirs(args.augment_out)
        image *= 255
        cv2.imwrite(os.path.join(args.augment_out, os.path.basename(image_file)), image)
