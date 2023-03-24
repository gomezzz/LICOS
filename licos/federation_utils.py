from pathlib import Path
import time
import os
import torch

from utils import save_checkpoint


def update_central_model(
    rank, device, batch_idx, net, loss, best_loss, local_time, cfg
):
    """Updates the model on the ground

    Args:
        rank (int): index of rank
        device (str): cuda device name
        batch_idx (int): index of batch
        net (torch.model): trained model
        loss (float): current loss value
        best_loss (float): best achieved loss value
        local_time (float): local time of the rank in seconds
        cfg (DotMap): cfg of the run
    """

    # Check for lock file (semaphore)
    print(f"Rank {rank} waiting for model update.")
    while os.path.exists(".mpi_lock"):
        time.sleep(0.1)

    # Claim semaphore
    Path(".mpi_lock").touch()
    print(f"Rank {rank} acquired semaphore for model update.")

    # Get local model state dict
    local_sd = net.state_dict()

    # Check there is already a central model, otherwise start one
    if os.path.exists(cfg.save_path + ".pth.tar"):
        # Load the current central model
        central_model = torch.load(cfg.save_path + ".pth.tar", map_location=device)
        central_model_sd = central_model["state_dict"]

        # TODO in the future consider which model is newer etc.
        # central_local_time = central_model["local_time"]

        # Weight models by test set scores.
        loss_weight_sum = 1.0 / (loss.item() + best_loss.item())

        # Average with ours
        for key in local_sd:
            local_sd[key] = (loss.item() * loss_weight_sum) * local_sd[key].to(
                device
            ) + (best_loss.item() * loss_weight_sum) * central_model_sd[key].to(device)
    else:
        print(f"Rank {rank} is starting the first central model.")

    # Overwrite the central model
    save_checkpoint(
        {
            "batch_idx": batch_idx,
            "state_dict": local_sd,
            "loss": loss,
            "local_time": local_time,
        },
        False,
        filename=cfg.save_path + ".pth.tar",
    )

    # Release lock
    if os.path.exists(".mpi_lock"):
        os.remove(".mpi_lock")
    print(f"Rank {rank} released semaphore for model update.")

    net.load_state_dict(local_sd)
