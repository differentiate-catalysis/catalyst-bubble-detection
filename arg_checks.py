from typing import List, Tuple
from types import SimpleNamespace

always_need = ["mode"]
always_allowed = ["config"]

optional = {
    "gen_labels": [],
    "augment": [],
    "image2npy": ["jobs", "test_set", "train_set", "val_set", "prompt", "split", "patch_size", "overlap_size"],
    "gen_targets": ["prompt", "run_dir", "split", "save", "train_set", "test_set", "val_set"],
    "train": ["gamma", "save", "num_images", "mp", "world_size", "nr", "amp", "momentum", "checkpoint", "prompt", "ray_checkpoint_dir", "graph", "imagenet_stats", "stats_file", "video", "image_size", "simclr_checkpoint", "loss", "blocks", "aug", "transforms", "patience", "collect", "version", "data_workers", "model_version"],
    "evaluate": ["video", "test_dir", "mp", "imagenet_stats", "stats_file", "image_size", "simclr_checkpoint", "amp", "patch_size", "overlap_size", "num_patches", "patching_mode", "loss", "blocks", "transforms", "version", "slices", "clear_predictions", "model_version"],
    "apply": ["video", "video_dir", "test_dir", "mp", "imagenet_stats", "stats_file", "image_size", "simclr_checkpoint", "amp", "patch_size", "overlap_size", "num_patches", "patching_mode", "loss", "blocks", "transforms", "version", "mask", "model_version"],
    "metrics": ["test_dir", "mask"],
    "optimize": ["min_lr_hpo", "max_lr_hpo", "min_momentum_hpo", "max_momentum_hpo", "min_epochs_hpo", "max_epochs_hpo", "sampling_models_hpo", "optimizers_hpo", "min_patch_size_hpo", "max_patch_size_hpo", "min_batch_size_hpo", "max_batch_size_hpo", "transforms", "min_gamma_hpo", "max_gamma_hpo", "max_epochs", "mp", "jobs", "num_samples_hpo", "resume_hpo", "patch_size", "losses", "sampling_slices", "slices", "min_overlap_size", "max_overlap_size", "sampling_patching_modes", "model_version"],
    "stitch": ["test_dir", "patch_size", "overlap_size", "num_patches", "patching_mode", "image_size", "version", "overlay", "slices", "clear_predictions"]
}

needed = {
    "gen_labels": ["json_dir", "root"],
    "augment": ["transforms", "root", "augment_out"],
    "image2npy": ["root", "num_patches", "run_dir", "name", "slices", "num_patches", "patching_mode", "gpu"],
    "gen_targets": ["jobs", "name", "root", "patch_size"],
    "train": ["gpu", "opt", "lr", "epoch", "name", "root", "batch_size", "model", "detections_per_img"],
    "evaluate": ["root", "name", "gpu", "model", "detections_per_img"],
    "apply": ["root", "name", "gpu", "model", "detections_per_img"],
    "metrics": ["root", "name"],
    "optimize": ["root", "name", "gpu", "detections_per_img"],
    "stitch": ["root", "name"],
}

hpo_condense = ['lr', 'epoch', 'momentum', 'patch_size', 'batch_size', 'gamma', 'loss', 'opt', 'model', 'slices', 'patching_mode', 'overlap_size']

def accum_args(modes: List[str]) -> Tuple[List[str], List[str]]:
    """Accumulate needed arguments for the list of modes given
    Args:
        modes (List[str]): List of modes being used (options: gen_labels, augment, image2npy, gen_targets, train, evaluate, apply, metrics, optimize, stitch)
    Returns:
        Tuple[List[str], List[str]]: (Set of arguments that must be present, set of arguments that can be present)
    """
    needed_args = always_need
    optional_args = always_allowed
    for mode in modes:
        if mode in needed.keys():
            for arg in needed[mode]: needed_args.append(arg)
        if mode in optional.keys():
            for arg in optional[mode]: optional_args.append(arg)
    return list(set(needed_args)), list(set(optional_args))
        
def check_args(modes: List[str], changed: List[str], explicit: List[str]) -> None:
    """Check to make sure arguments are all being used, warn or raise if needed
    Args:
        modes (List[str]): List of modes being checked (options: gen_labels, augment, image2npy, gen_targets, train, evaluate, apply, metrics, optimize, stitch)
        changed (List[str]): List of arguments that have been changed (i.e. by a config file). Used to check if any defaults are still being used.
        explicit (List[str]): List of arguments that have been explicitly changed (i.e. via command line arguments.) These will warn if they are not being used.
        
    """
    needed, optional = accum_args(modes)
    #Check if any explicit aren't needed
    for arg in explicit:
        if arg not in needed and arg not in optional:
            print("WARNING: argument " + arg + " is not needed and will not have any effect.")
    #Check if anything will be using default values
    for arg in needed:
        if arg not in changed:
            print("WARNING: using default value for argument " + arg + ", which may have undesired results.")

def condense_args(args: SimpleNamespace, modes: List[str]) -> SimpleNamespace:
    is_hpo = 'optimize' in modes
    for arg in hpo_condense:
        if not is_hpo:
            if type(getattr(args, arg)) is list:
                if len(getattr(args, arg)) > 1:
                    raise ValueError('When not HPO run, only one argument must be provided for arugment ' + arg)
                else:
                    setattr(args, arg, getattr(args, arg)[0]) #make this no longer list
        else:
            if type(getattr(args, arg)) is not list:
                setattr(args, arg, [getattr(args, arg)])
    return args
