# catalyst-bubble-detection

The `catalyst-bubble-detection` project aims to use ML to detect bubbles in high-throughput microscope images of catalyst surfaces. This repo contains the full ML pipeline for training, testing, evaluating, and optimizing ML models for bubble detection. The goal of this work is to screen catalysts to choose the most optimal one for a given gaseous reaction (e.g. oxygen evolution and oxygen reduction reactions).

## Installation

```
pip install -r requirements.txt
```

The install installs PyTorch 1.8.2 and Torchvision 0.9.2 (latest LTS release), and other required packages.

## Data Setup

This package is designed to be used with labeled data from [Darwin V7 Labs](https://www.v7labs.com/). Darwin V7 saves the labels of annotated data in the form of V7 JSON. Data should be organized in `images` and `json` folders as such:

```
root/
├─ images/
│  ├─ image1.jpg
│  ├─ image2.jpg
├─ json/
│  ├─ image1.json
│  ├─ image2.json
```

Names for images and their associated labels must be the same, with the exception of the file extension. 

If using a 3D semantic segmentation model (i.e. `unet` or `fcdensenet`), images sorted lexicographically are expected to be in order for a volume. Separate volumes should be in separate subfolders. Images should be in a `Im` subdirectory, and labels should be in a `L` subdirectory. Label files should have an extra `_L` appeneded to their file name. For instance:

```
root/
├─ vol1/
│  ├─ Im/
│  │  ├─ zslice01.png
│  │  ├─ zslice02.png
│  │  ├─ zslice03.png
│  ├─ L/
│  │  ├─ zslice01_L.png
│  │  ├─ zslice02_L.png
│  │  ├─ zslice03_L.png
├─ vol2/
│  ├─ Im/
│  │  ├─ zslice01.png
│  │  ├─ zslice02.png
│  │  ├─ zslice03.png
│  ├─ L/
│  │  ├─ zslice01_L.png
│  │  ├─ zslice02_L.png
│  │  ├─ zslice03_L.png
```

## Usage

```
python main.py --mode gen_targets train --config configs/test_config.json
```

## Configurations

Configurations change the behavior and settings of the application, and can be set in two ways, via command line flags or via [configuration files](#configuration-files). The order of configuration precedence is:

1. Configurations set via command line flags
2. Configurations set via a configuration file
3. A set of default configurations (as shown in [All Configuration Parameters](#all-configuration-parameters))

### Key Configuration Parameters

Generally, configurations should contain some key parameters:

* **`root`**: The root directory containing the data to be used
* **`name`**: The name of the model to be used (the model file will be saved automatically to `saved/<model_name>.pth`)
* **`gpu`**: The GPU to use when running the model. Use -1 for CPU only training.
* **`model`**: The type of model to use. See [Available Models](#available-models) for a list of options.

Additionally, each run should have one or more modes given as a command line argument. Most commonly used modes include `'train'`, `'apply'`, `'evaluate'`, and `'optimize'`. See [Detailed Usage](#detailed-usage) for more information on available modes.

### All Configuration Parameters

```
  --root ROOT           Root directory for the dataset. Default: data/bubbles
  --lr [LR ...]         Learning rate for the optimizer. Default: 0.0001
  --batch_size [BATCH_SIZE ...]
                        Batch size. Default: 1
  --epoch [EPOCH ...]   Number of epochs to train for. Default: 10
  --gpu GPU             Device ordinal for CUDA GPU. Set to -1 to run on CPU only. Default: -1
  --amp                 Use mixed precision (reduces VRAM requirements). Default: False
  --opt [OPT ...]       Optimizer to train with. Either 'adam' or 'sgd'. Default: adam
  --momentum [MOMENTUM ...]
                        Momentum for SGD. Default: 0.9
  --test_dir TEST_DIR   Directory to test model on. Default: test
  --save SAVE           Directory to write saved model weights to. Subdirectories will be created for HPO runs. Default: saved
  --mode MODE [MODE ...]
                        Mode to run. Available modes include 'train', 'apply', 'evaluate', 'optimize', 'gen_labels', 'augment',
                        'gen_targets', 'image2npy', 'metrics', and 'stitch'
  --mp                  Whether or not to use multiprocessing. Only use if you are training with multiple GPUs or multiple nodes. Default:
                        False
  --nodes NODES         Number of nodes for multiprocessing. Only necessary if mp is set.
  --nr NR               Index of the current node's first rank. Only necessary if mp is set.
  --transforms [TRANSFORMS ...]
                        Which augmentations to use. Choose some combination of 'vertical_flip', 'horizontal_flip', and 'rotation'. Default:
                        ['rotation', 'horizontal_flip', 'vertical_flip']
  --model [MODEL ...]   Which model architecture to use. Available models include 'mask_rcnn', 'faster_rcnn', 'retina_net', and
                        'simclr_faster_rcnn'. Default: faster_rcnn
  --config CONFIG       Location of a full configuration file. This contains options for a given run.
  --json_dir JSON_DIR   Directory containing several JSON configuration files.
  --num_images NUM_IMAGES
                        Number of images to train on
  --checkpoint CHECKPOINT
                        Filepath to a set of saved weights (pth file) to load
  --no-prompt           Skip prompt for checkpoint loading. Default: False
  --num_samples NUM_SAMPLES
                        Number of trials to run for HPO (only relevant if mode is 'optimize')
  --resume_hpo          Whether or not to resume an HPO trial
  --jobs JOBS           Number of cores to use for parsl
  --data_workers DATA_WORKERS
                        Number of processes to run for dataloading
  --name NAME           Name of the model
  --graph               Whether or not to save a plot of the results
  --run_dir RUN_DIR     Directory to use for parsl
  --loss [LOSS ...]     Loss function to use.
  --version VERSION     Version of the network (for MatCNN compatibility)
  --patience PATIENCE   Use patience after how many epochs
  --image_size IMAGE_SIZE IMAGE_SIZE
                        Size of images processed as x y (necessary for evaluate re-stitching)
  --num_patches NUM_PATCHES [NUM_PATCHES ...]
                        Number of patches each slice is split into. Set to 0 for automatic choice based on patch and image size.
  --overlap_size [OVERLAP_SIZE ...]
                        Width/height of the overlap between each patch
  --slices [SLICES ...]
                        Number of slices to take at a time
  --no-overlay          On evaluate, create full stitched images rather than overlays (if applicable)
  --data-split SPLIT SPLIT SPLIT
                        3 floats for test validation train split of data in image2npy (in order test, validation, train)
  --collect             Whether or not to manually garbage collect each input
  --patching_mode [PATCHING_MODE ...]
                        Mode with which to patch images. Either "grid" or "random"
  --clear_predictions   Clear prediction patches during evaluate
  --tar                 Tar output when done
  --mask MASK           Mask to apply when getting metrics
  --train_set TRAIN_SET
                        Dataset to use only as train
  --test_set TEST_SET   Dataset to use only as test
  --val_set VAL_SET     Dataset to use only as validation
  --blocks BLOCKS       Number of encoding blocks to use in each architecture
  --patch_size [PATCH_SIZE ...]
                        Size of patches to use for target generation. Set to -1 to disable patching
  --data_split SPLIT SPLIT SPLIT
                        3 floats for test validation train split of data in image2npy (in order test, validation, train)
  --gamma [GAMMA ...]   Amount to decrease LR by every 3 epochs
  --imagenet_stats      Use ImageNet stats instead of dataset stats for normalization
  --stats_file STATS_FILE
                        JSON file to read stats from
  --video VIDEO         Video file to perform inference on
  --video_dir VIDEO_DIR
                        Directory containing videos. Recursively searches for videos
  --simclr_checkpoint SIMCLR_CHECKPOINT
                        Weights for SimCLR pre-trained ResNet
  --augment_out AUGMENT_OUT
                        Output directory for augment mode
```

### Configuration Files

Any flags can also be included in a configuration file. The configuration file is a JSON file with key-value pairs for each flag and its associated value. This can then be used with the `--config` flag at runtime. For instance, an example `maskrcnn_test.json` config may contain the following:

```
{
    "root": "data/pristine-full",
    "model": "mask_rcnn",
    "aug": true,
    "lr": 1e-04,
    "opt": "adam",
    "transforms": ["horizontal_flip", "gray_balance_adjust", "blur", "sharpness", "vertical_flip", "rotation"],
    "epochs": 15,
    "gpu": 0,
    "amp": true,
    "batch_size": 8,
    "patch_size": 500,
    "prompt": false,
    "jobs": 8,
    "name": "maskrcnn_test"
}
```

## Detailed Usage

The general workflow starts with preprocessing via generating target data, training the model, then evaluating or applying (e.g. running inference with) the model.

Generally, most configuration parameters for all modes needed for a specific run are added to one configuration file. We provide a sample configuration file in `configs/sample.json` that shows the usage of many of the different configurations. A run using this configuration would then look like:

```
python main.py --config configs/sample.json --mode <mode or list of modes>
```

For improved results, we recommend using hyperparameter optimization (HPO) to find the optimal parameters for each given run. A sample configuration file demonstrating the different options for HPO is shown in `configs/sample_HPO.json`, which can then be run using:

```
python main.py --config configs/sample_HPO.json --mode optimize
```

### Generate Target Images

The `gen_targets` mode is used to generate target data for model training: images and associated .npy files containing the associated bubble masks and boxes.

The `splits` configuration option, a list of 3 floats, is used to determine the [test, validation, train] splits of the data. It defaults to `[0.1, 0.2, 0.7]`. The images in each split are chosen randomly.

To designate specific images for use in the train, test, and validation sets, place these images in separate folders (each containing `images` and `json` folders), and use the `train_set`, `test_set`, and `val_set` configuration options. 

`gen_targets` uses multiprocessing for accelerated runs. The number of threads utilized can be specified using the `jobs` configuration option.

Images can also be patched using the `patch_size` configuration option during this step. This is 
very useful for large images that may not easily fit in available RAM or GPU VRAM. Not including a patch size, or setting the patch_size to -1, will not split the image into patches.

`gen_targets` will often prompt the user, for instance if previous generated data will be overwritten. This can be bypassed with the `--no-prompt` command line argument or setting the `prompt` configuration option to `False`.

For semantic segmentation, generating target data is done using `image2npy` rather than `gen_targets` (see [Miscellaneous Mode Options](#miscellaneous-mode-options)).

### Train Model

The `train` mode is used to train models. Necessary training parameters can be set through the `epoch`, `opt` (optimizer), `lr` (learning rate), and `batch_size` configuration options.

Other options for training parameters include `gamma`, `mp`, `amp`, `momentum`, `patience`, and `collect`.

Various train-time augmentations can be added to greater vary the inputted data. This can be added by a list of options to the `transforms` configuration option and include:

* `horizontal_flip`
* `vertical_flip`
* `rotation`
* `gray_balance_adjust` (changes brightness, contrast, and saturation)
* `blur` (uses a Gaussian blur)
* `sharpness`
* `saltandpepper` (adds salt and pepper noise)
* `log_gamma` (Log Gamma contrast adjustment)
* `clahe` (Contrast Limited Adaptive Histogram Equalization)
* `contrast_stretch`

Training can also be done starting with a previous checkpoint file with pretrained weights. The `checkpoint` configuration option enables the use of pretrained weights. An example of using a checkpoint is shown in `configs/sample_HPO.json`.

### Evaluate and Apply Model

The `evaluate` mode runs the model on the test set of images, and outputs the resulting loss, Intersection over Union (IoU), and Mean Average Percision (mAP) scores. `evaluate` is effectively a call to `apply` followed by a call to `metrics`.

The `apply` mode runs the model on a set of unlabeled images. This mode can also be used to apply the model to a video file, where the model is run on each frame separately. Use the `video` configuration option to run on one video, or `video_dir` to run on a directory of videos.

To only obtain metrics for generated outputs, use the `metrics` mode.

### Perform Hyperparameter Optimization

Hyperparameter optimization (HPO) enables the utility to find a combination of parameters, such as learning rate, number of epochs, and batch size, that provide the best model performance. HPO is run using the `optimize` mode, and requires minimum and maximum values to search between for some parameters, and a list of options to choose from for others.

When creating a HPO configuration, the following arguments should have two values for the minimum and maximum values respectively:
* `lr`
* `epoch`
* `momentum`
* `patch_size`
* `batch_size`
* `gamma`
* `loss`

The `model` and `opt` arguments should be a list of different options for models and optimizers, respectively. Augmentations will be sampled from all augmentations listed in the `transforms` option.

The best option will be selected based on the best mAP score. The script generates a subdirectory within the `saved` directory with the best model stored as `best.pth`, and the associated configuration file saved as `best.json`.

### Miscellaneous Mode Options

Various other modes are available as well:

* `gen_labels`: Generate labels for use in semantic segmentation from V7Labs instance JSON files.
* `augment`: Apply augmentations to images and export to the directory specified in the `augment_out` configuration option.
* `image2npy`: The analog of `gen_targets` for semantic segmentation. The first two dimensions for each volume is set through `patch_size`, and the third dimension is set through `slices`.
* `stitch`: For use primarily in semantic segmentation, stitches patches together to create a full image.

## Available Models

### Instance Segmentation

```
mask_rcnn
faster_rcnn
retina_net
simclr_faster_rcnn
```

`mask_rcnn` and `faster_rcnn` both also support [version 1 and version 2 weights](https://pytorch.org/vision/master/models.html#instance-segmentation). Use the `model_version` arguments to switch between models. Version 1 is used by default for backwards compatability. 

### Semantic Segmentation

```
unet
fcdensenet
unet2d
```

Unet and FCDenseNet are 3D implementations.
