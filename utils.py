import shutil

import os 
import toml

import torch
import torch.nn as nn

from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models


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
    assert len(config["patch_size"]) == 2 and all(x > 0 for x in config["patch_size"]), "Patch size must have two positive elements"

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
        assert os.path.exists(config["checkpoint"]), f"Checkpoint {config['checkpoint']} does not exist"
        assert os.path.splitext(config["checkpoint"])[1] in [".pt", ".pth"], f"Checkpoint {config['checkpoint']} has an invalid extension"