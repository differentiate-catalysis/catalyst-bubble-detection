from io import TextIOWrapper
import os
import time
import sys
import gc
import json
import shutil
from types import SimpleNamespace
import argparse

import torch
from torch.nn.modules.module import Module
from torch.nn.modules.loss import _Loss
import torch.nn as nn
from torch.utils.data import DataLoader
from torchnet.meter import ConfusionMeter

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from MatCNN.metrics import metrics, bf1score, masks_mapping, bde
from MatCNN.utils import AverageMeter, Dataset
from MatCNN.models import get_model
from MatCNN.losses import loss_mappings

def stitch(image_size, num_patches, patch_size, patch_data, out_path, mode):
    concat = Image.new('RGBA', tuple([image_size[1], image_size[0]]))

    if num_patches[0] == 0:
        if not patch_size:
            raise ValueError("patch_size required if num_patches not provided")
        num_patches[0] = image_size[1] // patch_size + 1
        num_patches[1] = image_size[0] // patch_size + 1
        if (patch_size * num_patches[0] - image_size[1]) // (num_patches[0] - 1) + 1 < 20: #Force a minimum overlap of 20
            num_patches[0] += 1
        if (patch_size * num_patches[1] - image_size[0]) // (num_patches[1] - 1) + 1 < 20:
            num_patches[1] += 1
    overlap_x = (patch_size * num_patches[0] - image_size[1]) // (num_patches[0] - 1) + 1 if num_patches[0] != 1 else 0
    fill_y = image_size[0] // num_patches[1]
    overlap_y = (patch_size * num_patches[1] - image_size[0]) // (num_patches[1] - 1) + 1 if num_patches[1] != 1 else 0
    if overlap_x % 2 == 1: overlap_x += 1
    if overlap_y % 2 == 1: overlap_y += 1
    fill_x = (patch_size * num_patches[0] - overlap_x * (num_patches[0] - 1)) // num_patches[0]
    fill_y = (patch_size * num_patches[1] - overlap_y * (num_patches[1] - 1)) // num_patches[1]
    step_x = patch_size - overlap_x
    step_y = patch_size - overlap_y
    x_need_extra = (patch_size * num_patches[0] - overlap_x * (num_patches[0] - 1)) % num_patches[0]
    y_need_extra = (patch_size * num_patches[1] - overlap_y * (num_patches[1] - 1)) % num_patches[1]
    for k in range(num_patches[0]):
        for j in range(num_patches[1]):
            this_patch = patch_data[j * num_patches[0] + k].convert(mode='RGBA')
            patch_np = np.asarray(this_patch)
            patch_min_j = j * step_y
            patch_min_k = k * step_x
            min_j = fill_y * j - patch_min_j
            max_j = min_j + fill_y
            if (y_need_extra - 1) <= j: max_j += 1
            min_k = fill_x * k - patch_min_k
            max_k = min_k + fill_x
            if (x_need_extra - 1) <= k: max_k += 1
            cropped_np = patch_np[min_j : max_j, min_k : max_k, :]
            cropped_im = Image.fromarray(cropped_np)
            #print(k * patch_size, j * patch_size)
            concat.paste(cropped_im, (k * fill_x, j * fill_y))
    concat.save(out_path)

def sort_names(name):
    splits = name.replace('-', '_').split('_')
    splits[-1] = splits[-1][:-4]
    sort_list = []
    for part in splits:
        try:
            sort_list.append(int(part))
        except ValueError:
            sort_list.append(part)
    return tuple(sort_list)

def gen_full(label_dir, pred_dir, save_dir, class_num, slices, image_size, num_patches, patch_size, mode):

    # Get lists of label names and prediction names. // Only image file in prediction folder
    #if overlay: label_names = sorted([f for f in os.listdir(label_dir) if not f[0] == '.'], key = lambda f: sort_names(f))
    pred_names = sorted([f for f in os.listdir(pred_dir) if f[-3:] == 'png'], key = lambda f: sort_names(f))
    start = time.time() 
    # Go through each slice
    patches_per_img = num_patches[0] * num_patches[1]
    with tqdm(total = len(pred_names) // patches_per_img) as pbar:
        for i in range(slices):
            # Iterate through list of labels and list of predictions.
            output_images = []
            for file_num in range(len(pred_names) // slices):


                # Load prediction into a PIL image object .
                prediction = Image.open(os.path.join(pred_dir, pred_names[slices*file_num+i]))
                # If there are two classes, convert prediction image to numpy array of ones and zeroes corresponding
                # to each class. If there are more than two classes, convert prediction image to numpy array with
                # entries ranging to 0 to (class_num - 1) corresponding to each class. If number of classes provided
                # is less than 2, abort overlay process.
                if class_num == 2:
                    # Convert image from current format to 1-bit black and white format.
                    prediction = prediction.convert(mode='1')
                    # Convert image to array
                    pred_array = np.asarray(prediction) #.astype(dtype=label.dtype)
                elif class_num > 2:
                    # Convert image from current format to 8-bit greyscale format
                    prediction = prediction.convert(mode='L')
                    # Convert image to array with values potentially ranging from 0 to 255.
                    pred_array = np.asarray(prediction)

                    # Check if prediction array has values ranging up to 255. This attempts to handle how different
                    # image types are converted to numpy arrays. If this is the case, divide each entry so that they
                    # range from 0 to (class_num - 1). NOTICE: this process may vary based on label format.
                    if np.any(pred_array > class_num):
                        pred_array = pred_array / (255 / (class_num - 1))
                        '''
                        pred_array = pred_array.astype(dtype=label.dtype)
                    else:
                        pred_array = pred_array.astype(dtype=label.dtype)
                        '''
                else:
                    print('Number of Classes Provided Invalid!')
                    sys.exit(1)

                if mode == 'overlay' or mode == 'label':

                    if len([f for f in os.listdir(label_dir) if not f[0] == '.']) != len([f for f in os.listdir(pred_dir) if f[-3:] == 'png']) / slices:
                        print('Label and Prediction Directories Have Different Number of Files!')
                        return

                    # UINT8 corresponds to integer ranging from 0 to 255, as required by RGB image format.
                    label = np.load(os.path.join(label_dir, sorted([f for f in os.listdir(label_dir) if not f[0] == '.'], key = lambda f: sort_names(f))[file_num])).astype(dtype=np.uint8)
                    label = np.transpose(label, (2, 0, 1))


                    # Check if the shape of the label array and prediction array differ.
                    if label[i].shape != pred_array.shape:
                        print(label.shape)
                        print(pred_array.shape)
                        print('Dimension of Label and Prediction Differ!')
                        sys.exit(1)

                    # Create new array by stacking label three times and scale it range up to 255.
                    # This is done so we may convert directly to 'RGB' format.
                    correct_array = np.dstack((label[i], label[i], label[i]))
                    correct_array *= int(255 / (class_num - 1))

                    # Create image from layered array and convert it to 'RGBA' format. We will need the alpha
                    # channel to adjust transparency of the image.
                    correct_image = Image.fromarray(correct_array, mode='RGB').convert(mode='RGBA')

                    if mode == 'overlay':

                        # If there are two classes, create an image of green false positive pixels and an image of
                        # pink false negative pixels to overlay on the label image created above.
                        if class_num == 2:
                            # Determine if if an entry of the prediction array is greater than the corresponding entry
                            # of the label array. This indicates a false positive pixel, which will be mapped to a '1'
                            # at this position. Non-false positives will be mapped to a '0' elsewhere.
                            false_pos_array = np.greater(pred_array, label[i]).astype(dtype=np.uint8)

                            # Determine if if an entry of the prediction array is less than the corresponding entry
                            # of the label array. This indicates a false negative pixel, which will be mapped to a '1'
                            # at this position. Non-false negatives will be mapped to a '0' elsewhere.
                            false_neg_array = np.less(pred_array, label[i]).astype(dtype=np.uint8)

                            # Scale both the false positive and false negative arrays so to contain entries of 0 and 255.
                            false_pos_array *= 255
                            false_neg_array *= 255

                            # Stack arrays by the following: zero array of correct size, false positive array, zero
                            # array, and false positive array. This creates green image where non-false positive pixels
                            # are transparent upon converting to 'RGBA' image format.
                            false_pos_array = np.dstack((np.zeros_like(false_pos_array), false_pos_array,
                                                        np.zeros_like(false_pos_array), false_pos_array))

                            # Stack arrays by the following: false negative array, zero array of correct size, false
                            # negative array, and false negative array. This creates pink image where non-false negative
                            # pixels are transparent upon converting to 'RGBA' image format.
                            false_neg_array = np.dstack((false_neg_array, np.zeros_like(false_neg_array),
                                                        false_neg_array, false_neg_array))

                            # Create both false positive and false negative images in 'RGBA' format
                            false_pos_image = Image.fromarray(false_pos_array, mode='RGBA')
                            false_neg_image = Image.fromarray(false_neg_array, mode='RGBA')

                            # Overlay false positive and false negative image with label image by their alpha channels
                            # to retain transparency. This creates finalized overlay image.
                            overlay_image = Image.alpha_composite(correct_image, false_pos_image)
                            overlay_image = Image.alpha_composite(overlay_image, false_neg_image)

                        # If there are more than two classes, create an image of pink misidentified pixels to overlay
                        # onto label image.
                        else:
                            # Determines where prediction array does not equal label array and outputs a '1' at that
                            # position. Correctly identified pixels are mapped to '0' elsewhere.
                            false_array = np.not_equal(pred_array, label[i]).astype(dtype=np.uint8)

                            # Scale array to have entries 0 and 255 for conversion to 'RGBA' format
                            false_array *= 255

                            # Stack arrays as mentioned above to create pink image whose correctly identified pixels
                            # are transparent, and create image in 'RGBA' format from this layered array.
                            false_array = np.dstack((false_array, np.zeros_like(false_array), false_array, false_array))
                            false_image = Image.fromarray(false_array, mode='RGBA')

                            # Overlay this image with the label image by their alpha channels to retain transparency.
                            overlay_image = Image.alpha_composite(correct_image, false_image)

                        # Add the image to the list
                        output_images.append(overlay_image)
                    else:
                        output_images.append(correct_image)
                elif mode == 'full_image':
                    output_images.append(prediction)

            patches_per_img = num_patches[0] * num_patches[1]
            for file_num in range(0, len(pred_names) // slices, patches_per_img):
                split = sorted([f for f in os.listdir(pred_dir) if f[-3:] == 'png'], key = lambda f: sort_names(f))[slices*file_num+i][:-4].split('_')
                name = split[0] + '_' + str(int(split[1])+i)
                if mode == 'overlay': name = name + '_overlay.png'
                else: name = name + '.png'
                stitch(image_size, num_patches, patch_size, output_images[file_num:file_num + patches_per_img], os.path.join(save_dir, name), 'crop')
                pbar.update(1)
    print('Wrote %d images' % (len(pred_names) // patches_per_img))

    # Compute and display total time taken for each overlay, and indicate that it was successfully saved.
    end = time.time() - start
    print('Time Taken:    %.3f seconds' % end, '', sep='\n')


def get_metrics(pred_dir, label_dir, stats_file=None, indiv_file=None, losses=None, mask_name=None):
    full_conf_meter = ConfusionMeter(2)

    pred_names = sorted([f for f in os.listdir(pred_dir) if f[-3:] == 'png'], key = lambda f: sort_names(f))
    label_names = sorted([f for f in os.listdir(pred_dir) if f[-3:] == 'png'], key = lambda f: sort_names(f))
    pairs = zip(pred_names, label_names)
    bf1_list = []
    bde_list = []
    files_dict = {}

    metric_names = ('acc', 'iou', 'precision', 'recall', 'class_iou')

    for pred_file, label_file in tqdm(pairs, total=len(pred_names)):
        this_dict = {}
        pred_array = np.asarray(Image.open(os.path.join(pred_dir, pred_file))).astype(np.uint8)
        label_array = np.asarray(Image.open(os.path.join(label_dir, label_file))).astype(np.uint8)
        if len(pred_array.shape) == 3:
            pred_array = np.transpose(pred_array, (2, 0, 1))
            max_x = np.argmin(np.transpose(pred_array, (0, 2, 1))[3])
            max_y = np.argmin(pred_array[3])
            if max_x == 0:
                max_x = pred_array.shape[1]
            if max_y == 0:
                max_y = pred_array.shape[2]
            pred_array = pred_array[0, :max_x, :max_y]
        else:
            max_x = pred_array.shape[1]
            max_y = pred_array.shape[0]
        if pred_array.max() > 1:
            pred_array = pred_array // 255
        else:
            pred_array = pred_array // 1 #Need to do to make writeable for some reason
        if len(label_array.shape) == 3:
            label_array = np.transpose(label_array, (2, 0, 1))
            label_array = label_array[0, :max_x, :max_y]
        else:
            label_array = label_array[:max_x, :max_y]
        if label_array.max() > 1:
            label_array = label_array // 255
        else:
            label_array = label_array // 1

        if mask_name:
            mask, mask_contour = masks_mapping[mask_name]([max_x, max_y])
        else:
            mask, mask_contour = None, None

        bf1, precision, recall = bf1score(pred_array, label_array, 2, 4, mask=mask, mask_contour=mask_contour)
        #Apply mask for confusion meter
        if mask is not None:
            pred_array *= mask.astype('uint8')
            label_array *= mask.astype('uint8')
        bde_val = bde(pred_array, label_array)
        bde_list.append(bde_val)
        file_conf_meter = ConfusionMeter(2)
        file_conf_meter.add(torch.from_numpy(pred_array).view(-1), torch.from_numpy(label_array).view(-1))
        this_dict = dict(zip(metric_names, metrics(file_conf_meter.value(), verbose=False)))
        this_dict['bf1'], this_dict['Bprecision'], this_dict['Brecall'], this_dict['bde'] = bf1, precision, recall, bde_val
        full_conf_meter.add(torch.from_numpy(pred_array).view(-1), torch.from_numpy(label_array).view(-1))
        bf1_list.append(bf1)
        files_dict[pred_file] = this_dict

    full_conf_mat = full_conf_meter.value()
    # Obtain Accuracy, IoU, Class Accuracies, and Class IoU from metrics function
    acc, iou, precision, recall, class_iou = metrics(full_conf_mat, verbose=True)
    bf1_avg = np.nanmean(np.array(bf1_list), axis = 0)
    print('loss: %.5f, accuracy: %.5f, mIU: %.5f, BF1 Score: %.5f, BDE: %.5f' % (losses.avg if losses else 0, acc, iou, np.mean(bf1_avg), np.nanmean(bde_list)))
    if stats_file:
        stats_file.write('precision:' + str(np.round(precision, 5)) + str(precision.mean()) + '\n')
        stats_file.write('recall:' + str(np.round(recall, 5)) + str(recall.mean()) + '\n')
        stats_file.write('IoU:' + str(np.round(iou, 5)) + '\n')
        f1_score = (2 * precision * recall) / (precision + recall)
        stats_file.write('BF1 Score:' + str(np.round(bf1_avg, 5)) + str(bf1_avg.mean()) + '\n')
        stats_file.write('BDE:' + str(np.round(np.nanmean(bde_list), 5)) + '\n')
        stats_file.write('loss: %.5f, accuracy: %.5f, mIU: %.5f, class IoU: %.5f' % (losses.avg if losses else 0, acc, iou, class_iou))
    if indiv_file:
        json.dump(files_dict, indiv_file, cls=NumpyEncoder, indent=2)

class NumpyEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def evaluate(model: Module, criterion: _Loss, validloader: DataLoader, test_flag: bool = False, gpu: int = 0, save_dir: str = None, stats_file: TextIOWrapper = None, collect: bool = False, use_amp: bool = False):
    if gpu != -1:
        torch.cuda.empty_cache()
        device = torch.device('cuda', gpu)
    else:
        device = torch.device('cpu')
    losses = AverageMeter('Loss', ':.5f')

    # Define confusion matrix via built in PyTorch functionality
    if not test_flag:
        conf_meter = ConfusionMeter(2)
    with torch.no_grad():
        model.eval()
        for t, (inputs, labels, names) in enumerate(tqdm(validloader)):
            inputs, labels = inputs.to(device), labels.to(device).long()
            with torch.cuda.amp.autocast(enabled=(gpu != -1 and use_amp)):
                outputs = model(inputs)
                if labels is not None:
                    loss = criterion(outputs, labels)
                    losses.update(loss.item(), inputs.size(0))
            if test_flag:
                predictions = outputs.cpu().argmax(1)
                predictions_np = predictions.numpy()
                transposed = np.transpose(predictions_np, (0, 3, 1, 2))

                for i in range(predictions.shape[0]):
                    for j in range(transposed.shape[1]):
                        plt.imsave('%s/%s_%s.png' % (save_dir, names[i][:-4], str(j)), transposed[i, j].squeeze(), cmap='gray')
                        #plt.imshow(predictions[i].squeeze(), cmap='gray')
            if not test_flag:
                conf_meter.add(outputs.permute(0, 2, 3, 4, 1).contiguous().view(-1, 2), labels.view(-1))
            if collect:
                del inputs
                del labels
                del outputs
                gc.collect()
                if torch.cuda.is_available() and gpu != -1:
                    torch.cuda.empty_cache()
        if not test_flag:
            print('--- validation result ---')
            conf_mat = conf_meter.value()

            # Obtain Accuracy, IoU, Class Accuracies, and Class IoU from metrics function
            acc, iou, precision, recall, class_iou = metrics(conf_mat, verbose=test_flag)
            if not test_flag:
                print('loss: %.5f, accuracy: %.5f, mIU: %.5f' % (losses.avg, acc, iou))
                print('precision:', np.round(precision, 5))
            else:
                print('loss: %.5f, accuracy: %.5f, mIU: %.5f' % (losses.avg, acc, iou))
                if stats_file:
                    stats_file.write('precision:' + str(np.round(precision, 5)) + str(precision.mean()) + '\n')
                    stats_file.write('recall:' + str(np.round(recall, 5)) + str(recall.mean()) + '\n')
                    stats_file.write('IoU:' + str(np.round(iou, 5)) + '\n')
                    f1_score = (2 * precision * recall) / (precision + recall)
                    stats_file.write('BF1 Score:' + str(np.round(f1_score, 5)) + str(f1_score.mean()) + '\n')
                    stats_file.write('loss: %.5f, accuracy: %.5f, mIU: %.5f, class IoU: %.5f' % (losses.avg, acc, iou, class_iou))

            # Define second entry of precision vector as Soma accuracy
            class_precision = precision[1]
            return losses.avg, acc, iou, class_precision, class_iou
        else:
            return losses

def run_apply(args: SimpleNamespace):
    test_dir = os.path.join(args.root, args.name, args.test_dir)
    if not (os.path.isdir(os.path.join(test_dir, 'volumes_np')) and os.path.isdir(os.path.join(test_dir, 'labels_np'))):
        print("Skipping apply, no test data")
        return
    if os.path.isfile(os.path.join(test_dir, 'data_attributes.json')):
        with open(os.path.join(test_dir, 'data_attributes.json'), 'r') as fp:
            data_attributes = json.load(fp)
        args.patch_size = data_attributes['patch_size']
        args.overlap_size = data_attributes['overlap_size']
        args.image_size = data_attributes['image_size']
        args.num_patches = data_attributes['num_patches']
        args.patching_mode = data_attributes['patching_mode']
    if not (hasattr(args, 'patch_size') and args.patch_size):
        if not hasattr(args, 'overlap_size'):
            raise ValueError("One of patch_size or overlap_size required")
        args.patch_size = (args.overlap_size * (args.num_patches[0] - 1) + args.image_size[0]) // args.num_patches[0]
        if args.patch_size % 2 == 1:
            args.patch_size -= 1
    elif args.patch_size % 2 == 1:
        raise ValueError("Patch size must be even!")
    torch.manual_seed(0)

    if args.mp and args.gpu > -1:
        gpu = args.gpu - 1
    else:
        gpu = args.gpu

    if torch.cuda.is_available() and gpu > -1:
        criterion = loss_mappings[args.loss].cuda()
    else:
        if gpu > -1:
            print('No CUDA capable GPU detected. Using CPU...')
        else:
            print('Configuration has GPU set to -1. Using CPU...')
        criterion = loss_mappings[args.loss]
    if torch.cuda.is_available() and gpu > -1:
        device = torch.device('cuda', gpu)
        model = get_model(args.model, args.blocks)
        model.to(device)
    else:
        model = get_model(args.model, args.blocks)
    test_set = Dataset(test_dir, False, False, gpu, args.transforms)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=0, drop_last=False)

    model_dir = 'saved/%s.pth' % (args.name)#, args.version)
    model.load_state_dict(torch.load(model_dir)['model_state_dict'])
    if not os.path.isdir('saved'):
        os.makedirs('saved')
    save_dir = '%s/predictions/%s_%s' % (test_dir, args.name, args.version)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    loss = evaluate(model, criterion, validloader=test_loader, test_flag=True, gpu=gpu, save_dir=save_dir)
    return loss

def run_stitch(args: SimpleNamespace):
    test_dir = os.path.join(args.root, args.name, args.test_dir)
    if os.path.isfile(os.path.join(test_dir, 'data_attributes.json')):
        with open(os.path.join(test_dir, 'data_attributes.json'), 'r') as fp:
            data_attributes = json.load(fp)
        args.patch_size = data_attributes['patch_size']
        args.overlap_size = data_attributes['overlap_size']
        args.image_size = data_attributes['image_size']
        args.num_patches = data_attributes['num_patches']
        args.patching_mode = data_attributes['patching_mode']
    if not (hasattr(args, 'patch_size') and args.patch_size):
        if not hasattr(args, 'overlap_size'):
            raise ValueError("One of patch_size or overlap_size required")
        args.patch_size = (args.overlap_size * (args.num_patches[0] - 1) + args.image_size[0]) // args.num_patches[0]
        if args.patch_size % 2 == 1:
            args.patch_size -= 1
    elif args.patch_size % 2 == 1:
        raise ValueError("Patch size must be even!")
    save_dir = '%s/predictions/%s_%s' % (test_dir, args.name, args.version)
    full_pred_dir = '%s/full_predictions/%s_%s' % (test_dir, args.name, args.version)
    overlay_dir = '%s/overlays/%s_%s' % (test_dir, args.name, args.version)
    labels_dir = os.path.join(test_dir, 'labels_np')
    if not os.path.isdir('%s/predictions' % test_dir):
        os.makedirs('%s/predictions' % test_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if os.path.isdir(labels_dir) and args.overlay:
        if args.patching_mode == 'grid':
            if not os.path.isdir(overlay_dir):
                os.makedirs(overlay_dir)
            output_dir = overlay_dir
            print("Generating overlays:")
            gen_full(labels_dir, save_dir, output_dir, 2, args.slices, args.image_size, args.num_patches, args.patch_size, 'overlay')
        else:
            print("Can't generate full images or overlays from randomly sampled data")
    if not os.path.isdir(full_pred_dir):
        os.makedirs(full_pred_dir)
    output_dir = full_pred_dir
    if not len(os.listdir(save_dir)) == 0:
        print("Stitching predictions: ")
        gen_full(labels_dir, save_dir, output_dir, 2, args.slices, args.image_size, args.num_patches, args.patch_size, 'full_image')
    if os.path.isdir(labels_dir):
        label_stitch_dir = '%s/labels_stitched/%s_%s' % (test_dir, args.name, args.version)
        if not os.path.isdir(label_stitch_dir):
            os.makedirs(label_stitch_dir)
        print("Stitching labels: ")
        gen_full(labels_dir, save_dir, label_stitch_dir, 2, args.slices, args.image_size, args.num_patches, args.patch_size, 'label')
    if args.clear_predictions:
        shutil.rmtree('%s/predictions' % test_dir)

def run_metrics(args: SimpleNamespace, loss: bool = None):
    test_dir = os.path.join(args.root, args.name, args.test_dir)
    stats_dir = os.path.join('saved/statistics', os.path.basename(args.root))
    if not os.path.isdir(stats_dir):
        os.makedirs(stats_dir)
    full_pred_dir = '%s/full_predictions/%s_%s' % (test_dir, args.name, args.version)
    if not os.path.isdir(full_pred_dir):
            return
    label_stitch_dir = '%s/labels_stitched/%s_%s' % (test_dir, args.name, args.version)
    if not os.path.isdir(label_stitch_dir):
            return
    stats_output_file = open(os.path.join(stats_dir, '%s.txt' % (args.name)), 'w')
    indiv_output = open(os.path.join(stats_dir, '%s_individual_files.json' % (args.name)), 'w')
    print("Calculating metrics: ")
    get_metrics(full_pred_dir, label_stitch_dir, stats_file=stats_output_file, indiv_file=indiv_output, losses=loss, mask_name=args.mask)
    stats_output_file.close()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', dest='pred_dir', type=str, help='Path to prediction directory')
    parser.add_argument('-l', dest='label_dir', type=str, help='Path to label directory')
    parser.add_argument('--stats', dest='stats_file', type=str, help='Path to stats file')
    parser.add_argument('--indiv', dest='indiv_file', type=str, help='Path to file to store individual results')
    parser.add_argument('--mask', dest='mask_name', type=str, help='Name of mask to use')
    
    args = parser.parse_args()

    required_attrs = ['pred_dir', 'label_dir']

    prompt_mode = False
    for arg in required_attrs:
        if getattr(args, arg) is None:
            prompt_mode = True
            break
    
    if prompt_mode:
        prompts = {
            'pred_dir': 'Enter path to prediction directory: ',
            'label_dir': 'Enter path to label directory: ',
            'stats_file': 'Enter path to stats file: ',
            'indiv_file': 'Enter path to file to store individual results: ',
            'mask_name': 'Enter name of mask to use: '
        }
        for var, prompt in prompts.items():
            value = input(prompt)
            if value == '':
                value = None
            setattr(args, var, value)

    if args.stats_file is not None:
        fp = open(args.stats_file, 'w')
    if args.indiv_file is not None:
        indiv_fp = open(args.indiv_file, 'w')
    get_metrics(args.pred_dir, args.label_dir, stats_file=fp, indiv_file=indiv_fp, mask_name=args.mask_name)
    if args.stats_file is not None:
        fp.close()
    if args.indiv_file is not None:
        indiv_fp.close()
    

    
