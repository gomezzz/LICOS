import shutil

import os

from dotmap import DotMap

import torch
import torch.nn as nn

from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models

from datetime import datetime


def get_savepath_str(cfg: DotMap) -> str:
    """Determines sets and returns the path this run is saved under.

    Args:
        cfg (DotMap): config of the run

    Returns:
        str: path
    """
    return (
        "results/"
        + cfg.model
        + "_qual="
        + str(cfg.model_quality)
        + "_raw="
        + cfg.raw_format
        + "_seed="
        + str(cfg.seed)
        + "_t="
        + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    )


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, cfg):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": cfg.learning_rate},
        "aux": {"type": "Adam", "lr": cfg.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")


def save_model_checkpoint_over_time(cfg, local_time, rank, state):
    """Save model over time.

    Args:
        cfg (DotMap): cfg of the run
        local_time (float): local time of the rank in seconds
        rank (int): index of rank
        state (dict): model state.
    """
    # Directory path to local time_checkpoints
    model_name = cfg.save_path.split("/")[-1].split(".")[0]
    checkpoint_time_dir = os.path.join(cfg.save_path, model_name + "_time_checkpoints")

    # Adding rank to model name
    model_name = model_name + "_rank_" + str(rank)

    # Create sub-directory
    os.makedirs(checkpoint_time_dir, exist_ok=True)

    print(f"Saving checkpoint at simulation time: {local_time}.")
    save_checkpoint(
        state=state,
        is_best=False,
        filename=checkpoint_time_dir
        + "/"
        + model_name
        + "_sim_time="
        + str(local_time)
        + ".pth.tar",
    )


def check_cfg(config):
    """
    Checks the validity of the config entries.

    Args:
        config (dict): A dictionary containing the configuration parameters.

    Raises:
        AssertionError: If any of the config entries are invalid or missing.
    """
    # Check if the model is valid
    assert config["model"] in image_models.keys(), "Invalid model"

    # Check if the dataset is specified
    assert config["dataset"], "Dataset is required"

    # Check if the epochs are positive
    assert config["epochs"] > 0, "Epochs must be positive"

    # Check if the simulation time is positive
    assert config["simulation_time"] > 0, "Simulation time must be positive"

    # Check if the learning rate is positive
    assert config["learning_rate"] > 0, "Learning rate must be positive"

    # Check if the num workers are non-negative
    assert config["num_workers"] >= 0, "Num workers must be non-negative"

    # Check if the lambda is positive
    assert config["lambda"] > 0, "Lambda must be positive"

    # Check if the batch size is positive
    assert config["batch_size"] > 0, "Batch size must be positive"

    # Check if the test batch size is positive
    assert config["test_batch_size"] > 0, "Test batch size must be positive"

    # Check if the aux learning rate is positive
    assert config["aux_learning_rate"] > 0, "Aux learning rate must be positive"

    # Check if the patch size has two elements and they are both positive
    assert len(config["patch_size"]) == 2 and all(
        x > 0 for x in config["patch_size"]
    ), "Patch size must have two positive elements"

    # Check if cuda is a boolean value
    assert isinstance(config["cuda"], bool), "Cuda must be a boolean value"

    # Check if pretrained is a boolean value
    assert isinstance(config["pretrained"], bool), "pretrained must be a boolean value"

    # Check if save is a boolean value
    assert isinstance(config["save"], bool), "Save must be a boolean value"

    # If seed is specified, check if it is an integer value
    if config.get("seed"):
        assert isinstance(config["seed"], int), "Seed must be an integer value"

    # If checkpoint is specified, check if it exists and it has a valid extension (.pt or .pth)
    if config.get("checkpoint"):
        assert os.path.exists(
            config["checkpoint"]
        ), f"Checkpoint {config['checkpoint']} does not exist"
        assert os.path.splitext(config["checkpoint"])[1] in [
            ".pt",
            ".pth",
        ], f"Checkpoint {config['checkpoint']} has an invalid extension"

    # Check model quality values
    assert config.model_quality > 0 and config.model_quality < 9

    # Check raw values
    assert config.raw_format in ["split", "merged"]
    assert isinstance(config.use_raw_data, bool)
    assert isinstance(config.raw_target_resolution_merged_m, float)
    assert isinstance(config.raw_train_test_split, float)
    assert config.raw_train_test_split >= 0 and config.raw_train_test_split <= 1.0
