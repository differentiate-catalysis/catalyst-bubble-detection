from typing import List, Tuple

always_need = ["mode"]
always_allowed = ["config"]

optional = {
    "gen_labels": [],
    "augment": [],
    "image2npy": ["jobs", "test_set", "train_set", "val_set", "prompt", "split", "patch_size", "overlap_size"],
    "gen_targets": ["prompt"],
    "train": ["gamma", "mp", "world_size", "nr", "amp", "momentum", "checkpoint", "prompt", "ray_checkpoint_dir", "graph", "imagenet_stats", "stats_file", "video", "image_size", "simclr_checkpoint", "loss", "blocks", "aug", "transforms", "patience", "collect", "version", "data_workers"],
    "evaluate": ["video", "test_dir", "mp", "imagenet_stats", "stats_file", "image_size", "simclr_checkpoint", "amp", "patch_size", "overlap_size", "num_patches", "patching_mode", "loss", "blocks", "transforms", "version", "slices", "clear_predictions"],
    "apply": ["video", "test_dir", "mp", "imagenet_stats", "stats_file", "image_size", "simclr_checkpoint", "amp", "patch_size", "overlap_size", "num_patches", "patching_mode", "loss", "blocks", "transforms", "version", "mask"],
    "metrics": ["test_dir", "mask"],
    "optimize": ["min_lr_hpo", "max_lr_hpo", "min_momentum_hpo", "max_momentum_hpo", "min_epochs_hpo", "max_epochs_hpo", "sampling_models_hpo", "optimizers_hpo", "min_patch_size_hpo", "max_patch_size_hpo", "min_batch_size_hpo", "max_batch_size_hpo", "transforms", "min_gamma_hpo", "max_gamma_hpo", "max_epochs", "mp", "jobs", "num_samples_hpo", "resume_hpo", "patch_size", "losses", "sampling_slices", "slices", "min_overlap_size", "max_overlap_size", "sampling_patching_modes"],
    "stitch": ["test_dir", "patch_size", "overlap_size", "num_patches", "patching_mode", "image_size", "version", "overlay", "slices", "clear_predictions"]
}

needed = {
    "gen_labels": ["json_dir", "root"],
    "augment": ["transforms", "root", "augment_out"],
    "image2npy": ["root", "num_patches", "run_dir", "name", "slices", "num_patches", "patching_mode", "gpu"],
    "gen_targets": ["jobs", "run_dir", "name", "root", "patch_size", "split", "save"],
    "train": ["gpu", "opt", "lr", "epoch", "name", "root", "num_images", "batch_size", "model", "save"],
    "evaluate": ["root", "name", "gpu", "model"],
    "apply": ["root", "name", "gpu", "model"],
    "metrics": ["root", "name"],
    "optimize": ["root", "name", "gpu"],
    "stitch": ["root", "name"],
}

def accum_args(modes: List[str]) -> Tuple[List[str], List[str]]:
    needed_args = always_need
    optional_args = always_allowed
    for mode in modes:
        if mode in needed.keys():
            for arg in needed[mode]: needed_args.append(arg)
        if mode in optional.keys():
            for arg in optional[mode]: optional_args.append(arg)
    return list(set(needed_args)), list(set(optional_args))
        
def check_args(modes: List[str], changed: List[str], explicit: List[str]) -> None:
    needed, optional = accum_args(modes)
    #Check if any explicit aren't needed
    for arg in explicit:
        if arg not in needed and arg not in optional:
            print("WARNING: argument " + arg + " is not needed and will not have any effect.")
    #Check if anything will be using default values
    for arg in needed:
        if arg not in changed:
            print("WARNING: using default value for argument " + arg + ", which may have undesired results.")
