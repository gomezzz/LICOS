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

        # Average local and central model weights
        # by weighting with the validation set scores.

        # Normalize weights
        local_model_weight = best_loss.item() / (best_loss.item() + loss.item())
        central_model_weight = loss.item() / (best_loss.item() + loss.item())

        # Average the weights
        for key in local_sd:
            local_sd[key] = local_model_weight * local_sd[key].to(device)
            local_sd[key] += central_model_weight * central_model_sd[key].to(device)

        # Save checkpoint over time if requested in the cfg
        if cfg.save_checkpoints_over_time:
            # Directory path to local time_checkpoints
            model_name = cfg.save_path.split("/")[-1].split(".")[0]
            checkpoint_time_dir = os.path.join(
                cfg.save_path, model_name + "_time_checkpoints"
            )
            print(
                checkpoint_time_dir
                + "/"
                + model_name
                + "_sim_time="
                + str(local_time)
                + ".pth.tar"
            )
            # Create sub-directory
            os.makedirs(checkpoint_time_dir, exist_ok=True)

            print(f"Saving checkpoint at simulation time: {local_time}.")
            save_checkpoint(
                {
                    "batch_idx": batch_idx,
                    "state_dict": local_sd,
                    "loss": loss,
                    "local_time": local_time,
                },
                False,
                filename=checkpoint_time_dir
                + "/"
                + model_name
                + "_sim_time="
                + str(local_time)
                + ".pth.tar",
            )
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
