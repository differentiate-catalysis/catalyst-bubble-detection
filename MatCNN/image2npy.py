import os
import shutil
import random
import json
import time

import torch
import torch.nn as nn
import numpy as np
#from scipy import signal
#from numpy.lib.stride_tricks import as_strided
from PIL import Image
from tqdm import tqdm
import parsl
from parsl import python_app
from parsl.config import Config
from parsl.executors.threads import ThreadPoolExecutor

@python_app
def random_select(dataset, slices, num_patches, subdir, index, files, namestart, destination, patch_size, acceptable_overlap, gpu=True):
    # Skip volumes that include a HandSeg z-slice
    for file in files:
        split = file.split('/')
        timestamp = split[-4]
        z_slice = int(split[-1][-7:-4])
        if 131 <= z_slice and z_slice <= 190 and timestamp == 'c62-66':
            return

    out_dir = os.path.join(dataset, destination)
    sample = []
    for file in files:
        image = Image.open(file)
        image = image.convert(mode = 'L')
        image_arr = np.asarray(image)[:, :, np.newaxis]
        sample.append(image_arr)
    # Transpose for the format for the CNN: c, y, x, z
    sample = np.transpose(sample, axes = (3, 1, 2, 0))
    volumes_out = os.path.join(out_dir, 'volumes_np')
    if not os.path.isdir(volumes_out):
        os.makedirs(volumes_out, exist_ok=True)
        with open(os.path.join(out_dir, 'data_attributes.json'), 'w') as fp:
            json.dump({
                'patch_size': patch_size,
                'image_size': sample.shape[1:3],
                'overlap_size': None,
                'num_patches': None,
                'patching_mode': 'random'
            }, fp, indent=2)
    full_x = sample.shape[2]
    full_y = sample.shape[1]
    if not patch_size:
        raise ValueError("patch_size required for random sampling!")
    available_origins = np.zeros((full_y, full_x), dtype='float64')
    available_origins[:(patch_size - 1) * -1, :(patch_size - 1) * -1] = 1
    origins = []
    if not acceptable_overlap:
        overlap = 0
    else:
        overlap = acceptable_overlap
        device = 'cuda' if gpu else 'cpu'
        if (patch_size - overlap * 2) ** 2 * num_patches > full_x * full_y:
            raise ValueError("%i %i x %i patches won't fit in %i x %i image with overlap %i" %(num_patches, patch_size, patch_size, full_x, full_y, overlap))
        filter = nn.Conv2d(1, 1, (patch_size - overlap) * 2 - 1, bias=False, padding=patch_size - overlap - 1).to(device=device)
        filter.weight.data = torch.ones_like(filter.weight, dtype=torch.float64)
    while not origins:
        attempts = 0
        for i in range(num_patches):
            if not acceptable_overlap:
                #Our exit strategy is continuously increasing overlap, so we'll just keep trying
                origin_options = np.argwhere(available_origins)

            else:
                #Our exit strategy is ensuring the number of options eliminated is less than what's needed to avoid problems
                start_time = time.monotonic()
                '''
                numpy version:
                padded = np.pad(available_origins, (patch_size - overlap - 1))
                sub_shape = ((patch_size - overlap) * 2 - 1, (patch_size - overlap) * 2 - 1)
                view_shape = tuple(np.subtract(padded.shape, sub_shape) + 1) + sub_shape
                options_lost = np.sum(as_strided(padded, view_shape, padded.strides * 2), axis=(2, 3)) * available_origins
                options_lost[options_lost == 0] = np.nan
                '''
                '''
                scipy version:
                options_lost = signal.convolve2d(available_origins, filter, mode='same')
                options_lost[available_origins == 0] == np.nan
                origin_options = np.argwhere(options_lost <= full_x * full_y / num_patches)
                '''
                print("Finding origin " + str(i))
                t_avail = torch.from_numpy(available_origins).to(device=device).unsqueeze(0).unsqueeze(0)
                options_lost = filter(t_avail).squeeze().detach()
                #options_lost = options_lost.numpy()
                options_lost[t_avail.squeeze() == 0] == float('nan')
                origin_options = torch.nonzero(options_lost <= full_x * full_y / num_patches).cpu().numpy()
                elapsed = time.monotonic() - start_time
            #if len(origin_options) != 0: print(len(origin_options))
            if len(origin_options) == 0:
                available_origins = np.zeros((full_y, full_x), dtype='float64')
                available_origins[:(patch_size - 1) * -1, :(patch_size - 1) * -1] = 1
                origins = []
                if not acceptable_overlap:
                    #Exponentially increase necessary overlap
                    overlap = overlap * 2 if overlap != 0 else 1
                    overlap = min(overlap, patch_size)
                #print("Failed to generate origins, trying again")
                #print(i)
                attempts += 1  
                break
            origin = origin_options[random.randint(0, len(origin_options) - 1)]
            available_origins[max([origin[0] - patch_size + overlap + 1, 0]):min([origin[0] + patch_size - overlap, full_y]), max([origin[1] - patch_size + overlap + 1, 0]):min([origin[1] + patch_size - overlap, full_x])] = 0
            origins.append(origin)
    for origin in origins:
        sample_out = sample[:, origin[0] : origin[0] + patch_size, origin[1] : origin[1] + patch_size, :]
        name = str(namestart) + '_' + str(int(index / slices)) + '_' + str(origin[0]) + '_' + str(origin[1]) + '.npy'
        np.save(os.path.join(volumes_out, name), sample_out)
    sample = []
    label_path = os.path.join(subdir, 'labels')
    for file in files:
        split_file = os.path.splitext(os.path.basename(file))
        filepath = os.path.join(label_path, split_file[0] + '.png')
        image = Image.open(filepath)
        image = image.convert(mode = 'L')
        image_arr = np.asarray(image)
        sample.append(image_arr)
    sample = np.transpose(sample, axes = (1, 2, 0))
    sample = sample / 255
    sample = np.array(sample, dtype=np.uint8)
    labels_out = os.path.join(out_dir, 'labels_np')
    if not os.path.isdir(labels_out):
        os.makedirs(labels_out, exist_ok=True)
    for origin in origins:
        sample_out = sample[origin[0] : origin[0] + patch_size, origin[1] : origin[1] + patch_size, :]
        np.save(os.path.join(labels_out, str(namestart) + '_' + str(int(index / slices)) + '_' + str(origin[0]) + '_' + str(origin[1]) + '.npy'), sample_out) 


@python_app
def grid_select(dataset, slices, num_patches, subdir, index, files, namestart, destination, patch_size=None, overlap_size=None):
    # Skip volumes that include a HandSeg z-slice
    '''
    for file in files:
        split = file.split('/')
        timestamp = split[-4]
        z_slice = int(split[-1][-7:-4])
        if 131 <= z_slice and z_slice <= 190 and timestamp == 'c62-66':
            return
    
    
    
    if destination == 'test':
        test = True
    '''
    test = False
    out_dir = os.path.join(dataset, destination)
    sample = []
    for file in files:
        image = Image.open(file)
        image = image.convert(mode = 'L')
        image_arr = np.asarray(image)[:, :, np.newaxis]
        sample.append(image_arr)
    # Transpose for the format for the CNN: c, y, x, z
    sample = np.transpose(sample, axes = (3, 1, 2, 0))
    volumes_out = os.path.join(out_dir, 'volumes_np')
    if not os.path.isdir(volumes_out):
        os.makedirs(volumes_out, exist_ok=True)
        with open(os.path.join(out_dir, 'data_attributes.json'), 'w') as fp:
            json.dump({
                'patch_size': patch_size,
                'image_size': sample.shape[1:3],
                'overlap_size': overlap_size,
                'num_patches': num_patches,
                'patching_mode': 'grid'
            }, fp, indent=2)
    full_x = sample.shape[2]
    full_y = sample.shape[1]
    if num_patches[0] == 0:
        if not patch_size:
            raise ValueError("patch_size required if num_patches not provided")
        num_patches[0] = full_x // patch_size + 1
        num_patches[1] = full_y // patch_size + 1
        if (patch_size * num_patches[0] - full_x) // (num_patches[0] - 1) + 1 < 20: #Force a minimum overlap of 20
            num_patches[0] += 1
        if (patch_size * num_patches[1] - full_y) // (num_patches[1] - 1) + 1 < 20:
            num_patches[1] += 1
    if not patch_size and overlap_size:
        overlap_x = overlap_size
        overlap_y = overlap_size
        patch_x = (overlap_size * (num_patches[0] - 1) + full_x) // num_patches[0]
        patch_y = (overlap_size * (num_patches[1] - 1) + full_y) // num_patches[1]
        if patch_x % 2 == 1:
            patch_x -= 1
        if patch_y % 2 == 1:
            patch_y -= 1
    elif patch_size:
        if patch_size % 2 == 1:
            raise ValueError("Patch size must be even!")
        patch_x = patch_size
        patch_y = patch_size
        overlap_x = (patch_size * num_patches[0] - full_x) // (num_patches[0] - 1) + 1 if num_patches[0] != 1 else 0
        overlap_y = (patch_size * num_patches[1] - full_y) // (num_patches[1] - 1) + 1 if num_patches[1] != 1 else 0
    else:
        raise ValueError("One of patch_size or overlap_size required")
    if overlap_x % 2 == 1: overlap_x += 1
    if overlap_y % 2 == 1: overlap_y += 1
    step_x = patch_x - overlap_x
    step_y = patch_y - overlap_y
    if not test:
        for j in range(num_patches[0]):
            for k in range(num_patches[1]):
                min_j = j * step_x
                max_j = min_j + patch_x
                min_k = k * step_y
                max_k = min_k + patch_y
                sample_out = sample[:, min_k : max_k, min_j : max_j, :]
                name = str(namestart) + '_' + str(int(index / slices)) + '_' + str(k) + '_' + str(j) + '.npy'
                np.save(os.path.join(volumes_out, name), sample_out)
    else:
        name = str(namestart) + '_' + str(int(index / slices))
        np.save(os.path.join(volumes_out, name), sample)
    sample = []
    label_path = os.path.join(subdir, 'labels')
    for file in files:
        split_file = os.path.splitext(os.path.basename(file))
        filepath = os.path.join(label_path, split_file[0] + '.png')
        image = Image.open(filepath)
        image = image.convert(mode = 'L')
        image_arr = np.asarray(image)
        sample.append(image_arr)
    sample = np.transpose(sample, axes = (1, 2, 0))
    sample = sample / 255
    sample = np.array(sample, dtype=np.uint8)
    labels_out = os.path.join(out_dir, 'labels_np')
    if not os.path.isdir(labels_out):
        os.makedirs(labels_out, exist_ok=True)
    if not test:
        for j in range(num_patches[0]):
            for k in range(num_patches[1]):
                min_j = j * step_x
                max_j = min_j + patch_x
                min_k = k * step_y
                max_k = min_k + patch_y
                sample_out = sample[min_k : max_k, min_j : max_j, :]
                np.save(os.path.join(labels_out, str(namestart) + '_' + str(int(index / slices)) + '_' + str(k) + '_' + str(j) + '.npy'), sample_out) 
    else:
        name = subdir[1:] + '_' + str(int(index / slices))
        np.save(os.path.join(labels_out, name), sample)

def gen_workers(args, splits, dataset, workers):
    patch_size = args.patch_size if hasattr(args, 'patch_size') else None
    overlap_size = args.overlap_size if hasattr(args, 'overlap_size') else None
    slices = args.slices
    num_patches = args.num_patches
    if type(num_patches) == int:
        num_patches = [num_patches, num_patches]
    elif len(num_patches) == 1:
        num_patches = [num_patches[0], num_patches[0]]
    print(num_patches)
    subdirs = []
    for root, dirs, files in os.walk(dataset, followlinks=True):
        if os.path.isdir(os.path.join(root, 'images')) and os.path.isdir(os.path.join(root, 'labels')):
            subdirs.append(root)
    subdirs = sorted(subdirs)
    names = [os.path.basename(path).replace("_", "") for path in subdirs]
    dirs_up = 0
    while len(names) != len(set(names)): #Check if duplicates in names
        dirs_up += 1
        for i in range(len(names)):
            subdir = subdirs[i]
            for iter in range(dirs_up):
                subdir = os.path.dirname(subdir)
            if subdir == '/': #We've gone all the way up to no avail
                names = [i for i in range(len(names))]
                break
            names[i] = os.path.basename(subdir)
    file_tuples = []
    for subdir_count in range(len(subdirs)):
        subdir = subdirs[subdir_count]
        # Get all the images from the subdirectory
        images = os.path.join(subdir, 'images')
        files = [os.path.join(images, file) for file in os.listdir(images) if os.path.isfile(os.path.join(images, file))]
        files.sort()
        for i in range(0, len(files) - slices + 1, slices):
            # Process the volume
            file_tuples.append((subdir, names[subdir_count] + '_' + str(i), files[i : i + slices]))
    random.seed(89955998)
    random.shuffle(file_tuples)
    for i in range(len(file_tuples)):
        subdir, name, files = file_tuples[i]
        progress = i / len(file_tuples)
        if progress < splits[0]:
            destination = os.path.join(args.name, 'test')
        elif progress < splits[0] + splits[1]:
            destination = os.path.join(args.name, 'validation')
        else:
            destination = os.path.join(args.name, 'train')
        if args.patching_mode == 'random':
            workers.append(random_select(args.root, slices, num_patches[0], subdir, i, files, name, destination, patch_size, overlap_size, gpu=(args.gpu != 0)))
        else:
            workers.append(grid_select(args.root, slices, num_patches.copy(), subdir, i, files, name, destination, patch_size=patch_size, overlap_size=overlap_size))

def convert_dir(args):
    dataset = args.root
    num_patches = args.num_patches
    jobs = args.jobs if hasattr(args, 'jobs') else 1
    local_threads = Config(
        executors=[
            ThreadPoolExecutor(
                max_threads=jobs,
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
    if args.test_set:
        test_out = os.path.join(args.root, args.name, 'test')
    else:
        test_out = os.path.join(dataset, args.name, 'test')
    if args.train_set:
        train_out = os.path.join(args.root, args.name, 'train')
    else:
        train_out = os.path.join(dataset, args.name, 'train')
    if args.val_set:
        valid_out = os.path.join(args.root, args.name, 'validation')
    else:
        valid_out = os.path.join(dataset, args.name, 'validation')
    if os.path.isdir(train_out):
        response = input('Train directory already exists! Wipe? (y/n) ').lower() if args.prompt else 'y'
        if response == 'y' or response == 'ye' or response == 'yes':
            shutil.rmtree(train_out)
    if os.path.isdir(test_out):
        response = input('Test directory already exists! Wipe? (y/n) ').lower() if args.prompt else 'y'
        if response == 'y' or response == 'ye' or response == 'yes':
            shutil.rmtree(test_out)
    if os.path.isdir(valid_out):
        response = input('Validation directory already exists! Wipe? (y/n) ').lower() if args.prompt else 'y'
        if response == 'y' or response == 'ye' or response == 'yes':
            shutil.rmtree(valid_out)
    if not os.path.isdir(train_out):
        os.makedirs(train_out)
    if not os.path.isdir(test_out):
        os.makedirs(test_out)
    if not os.path.isdir(valid_out):
        os.makedirs(valid_out)
    # Find all subdirectories with images and L directories inside
    
    workers = []
    if not (args.train_set or args.val_set or args.test_set):
        gen_workers(args, args.split, dataset, workers)
    if args.train_set:
        gen_workers(args, [0, 0, 1], os.path.join(args.root, args.train_set), workers)
    if args.val_set:
        gen_workers(args, [0, 1, 0], os.path.join(args.root, args.val_set), workers)
    if args.test_set:
        gen_workers(args, [1, 0, 0], os.path.join(args.root, args.test_set), workers)
    for i in tqdm(workers):
        result = i.result()
    parsl.clear()
