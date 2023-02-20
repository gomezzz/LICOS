import sys
import time
import os
from pathlib import Path

import torch
import numpy as np
import pykep as pk
import paseos
from mpi4py import MPI

from utils import (
    parse_args,
    save_checkpoint,
)
from create_plots import create_plots
from init_paseos import init_paseos
from train import train_one_batch, test_epoch, init_training


def constraint_func(paseos_instance: paseos.PASEOS, actors_to_track):
    """Constraint function for activitiy

    Args:
        paseos_instance (paseos): Local instance
        actors_to_track (Actor): Actors we want to communicate with

    Returns:
        Whether constraint is still met
    """

    # Update known actors
    local_t = paseos_instance.local_actor.local_time
    paseos_instance.emtpy_known_actors()
    for actor in actors_to_track:
        if paseos_instance.local_actor.is_in_line_of_sight(actor, local_t):
            paseos_instance.add_known_actor(actor)

    # Check constraints
    if paseos_instance.local_actor.temperature_in_K > (273.15 + 65):
        return False
    if paseos_instance.local_actor.state_of_charge < 0.1:
        return False

    return True


def decide_on_activity(
    paseos_instance: paseos.PASEOS,
    timestep,
    timestep_for_comms,
    time_in_standby,
    standby_period,
    time_since_last_update,
):
    """Heuristic to decide activitiy for the actor. Initiates a standby period of passed
    length when going into standby.

    Args:
        paseos_instance (paseos): Local instance

    Returns:
        activity,power_consumption
    """
    has_comm_window = False
    window_end = pk.epoch(
        paseos_instance.local_actor.local_time.mjd2000 + timestep_for_comms * pk.SEC2DAY
    )
    for actors in paseos_instance.known_actors.items():
        if paseos_instance.local_actor.is_in_line_of_sight(
            actors[1], epoch=paseos_instance.local_actor.local_time
        ) and paseos_instance.local_actor.is_in_line_of_sight(
            actors[1], epoch=window_end
        ):
            has_comm_window = True
            break
    if (
        has_comm_window
        and time_since_last_update > 900
        and len(paseos_instance.known_actors) > 0
        and paseos_instance.local_actor.state_of_charge > 0.1
        and paseos_instance.local_actor.temperature_in_K < 273.15 + 45
    ):
        return "Model_update", 10, 0
    elif (
        paseos_instance.local_actor.temperature_in_K > (273.15 + 40)
        or paseos_instance.local_actor.state_of_charge < 0.2
        or (time_in_standby > 0 and time_in_standby < standby_period)
    ):
        return "Standby", 5, time_in_standby + timestep
    else:
        # Wattage from 1605B https://www.amd.com/en/products/embedded-ryzen-v1000-series
        # https://unibap.com/wp-content/uploads/2021/06/spacecloud-ix5-100-product-overview_v23.pdf
        return "Training", 30, 0


def perform_activity(
    activity, power_consumption, paseos_instance, time_to_run, constraint_function
):
    paseos_instance.local_actor._current_activity = activity
    return_code = paseos_instance.advance_time(
        time_to_run,
        current_power_consumption_in_W=power_consumption,
        constraint_function=constraint_function,
    )
    if return_code > 0:
        raise ("Activity was interrupted. Constraints no longer true?")


def update_central_model(
    rank,
    device,
    batch_idx,
    net,
    loss,
    best_loss,
    local_time,
):
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
    if os.path.exists("central_model.pth.tar"):
        # Load the current central model
        central_model = torch.load("central_model.pth.tar", map_location=device)
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
        filename="central_model.pth.tar",
    )

    # Release lock
    if os.path.exists(".mpi_lock"):
        os.remove(".mpi_lock")
    print(f"Rank {rank} released semaphore for model update.")

    net.load_state_dict(local_sd)


def aggregate(rank_model, received_models, device):
    cw = 1 / (len(received_models) + 1)

    local_sd = rank_model.state_dict()
    for key in local_sd:
        local_sd[key] = cw * local_sd[key].to(device) + sum(
            [sd[key].to(device) * cw for i, sd in enumerate(received_models)]
        )

    # update server model with aggregated models
    rank_model.load_state_dict(local_sd)


def eval_test_set(
    rank,
    optimizer,
    batch_idx,
    net,
    criterion,
    test_losses,
    test_dataloader,
    local_time_at_test,
    paseos_instance,
    lr_scheduler,
    best_loss,
):
    # print(f"Rank {rank} - Evaluating test set")
    # print(f"Rank {rank} - Previous learning rate: {optimizer.param_groups[0]['lr']}")
    start = time.time()
    loss = test_epoch(rank, batch_idx, test_dataloader, net, criterion)
    test_losses.append(loss.item())
    local_time_at_test.append(paseos_instance._state.time)
    # print(f"Rank {rank} - Test evaluation took {time.time() - start}s")

    lr_scheduler.step(loss)
    # print(f"Rank {rank} - New learning rate: {optimizer.param_groups[0]['lr']}")

    is_best = loss < best_loss
    best_loss = min(loss, best_loss)
    return loss, is_best, best_loss


def main(argv):
    # Init
    rank = 0  # compute index of this node
    time_per_batch = 0.1 * 100  # estimated time per batch in seconds
    assert time_per_batch < 30, "For a high time per batch you may miss comms windows?"
    time_for_comms = 60
    time_in_standby = 0
    time_since_last_update = 0
    standby_period = 900  # how long to standby if necessary
    plot = False
    test_losses = []
    local_time_at_test = []
    constraint_function = lambda: constraint_func(paseos_instance, groundstations)
    args = parse_args(argv)
    paseos.set_log_level("INFO")
    device = "cuda:" + str(rank) if args.cuda and torch.cuda.is_available() else "cpu"
    args.pretrained = False

    # Init MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    other_ranks = [x for x in range(comm.Get_size()) if x != rank]
    print(f"Started rank {rank}, other ranks are {other_ranks}")
    sys.stdout.flush()

    # Remove any prior lock
    if rank == 0 and os.path.exists(".mpi_lock"):
        print("Removing old lock...")
        os.remove(".mpi_lock")

    if rank == 0 and os.path.exists("central_model.pth.tar"):
        print("Removing old model...")
        os.remove("central_model.pth.tar")

    # Init training
    (
        net,
        optimizer,
        aux_optimizer,
        criterion,
        train_dataloader,
        test_dataloader,
        lr_scheduler,
        last_epoch,
        train_dataloader_iter,
    ) = init_training(args, rank)

    print(f"Rank {rank} - Init training")
    sys.stdout.flush()

    # Init paseos
    paseos_instance, local_actor, groundstations = init_paseos(rank, comm.Get_size())

    print(f"Rank {rank} - Init PASEOS")
    sys.stdout.flush()

    if plot and rank == 0:
        plotter = paseos.plot(paseos_instance, paseos.PlotType.SpacePlot)

    # Training loop
    best_loss = float("inf")
    batch_idx = 0
    while batch_idx < args.epochs:
        if batch_idx % 100 == 0:
            print(
                f"Rank {rank} - Temperature[C]: "
                + f"{local_actor.temperature_in_K - 273.15:.2f},"
                + f"Battery SoC: {local_actor.state_of_charge:.2f}"
            )
            sys.stdout.flush()
            # print(f"PASEOS advancing time by {time_per_batch}s.")

        activity, power_consumption, time_in_standby = decide_on_activity(
            paseos_instance,
            time_per_batch,
            time_for_comms,
            time_in_standby,
            standby_period,
            time_since_last_update,
        )

        if activity == "Model_update":
            print(
                f"Rank {rank} will update with GS "
                + str(list(paseos_instance.known_actors.items())[0][0])
                + " at "
                + str(paseos_instance.local_actor.local_time)
            )
            perform_activity(
                activity,
                power_consumption,
                paseos_instance,
                time_for_comms,
                constraint_function,
            )
            print(f"Rank {rank} - Pre-aggregation test.")
            loss, is_best, best_loss = eval_test_set(
                rank,
                optimizer,
                batch_idx,
                net,
                criterion,
                test_losses,
                test_dataloader,
                local_time_at_test,
                paseos_instance,
                lr_scheduler,
                best_loss,
            )

            update_central_model(
                rank,
                device,
                batch_idx,
                net,
                loss,
                best_loss,
                paseos_instance._state.time,
            )
            time_since_last_update = 0

            print(f"Rank {rank} - Post-aggregation test.")
            loss, is_best, best_loss = eval_test_set(
                rank,
                optimizer,
                batch_idx,
                net,
                criterion,
                test_losses,
                test_dataloader,
                local_time_at_test,
                paseos_instance,
                lr_scheduler,
                best_loss,
            )
            # Push the time of last step slightly beyond to be distinguishable in plots
            local_time_at_test[-1] += 10
        elif activity == "Training":
            perform_activity(
                activity,
                power_consumption,
                paseos_instance,
                time_per_batch,
                constraint_function,
            )
            time_since_last_update += time_per_batch
            # Train one
            start = time.time()
            train_dataloader_iter = train_one_batch(
                rank,
                net,
                criterion,
                train_dataloader,
                train_dataloader_iter,
                optimizer,
                aux_optimizer,
                batch_idx,
                args.clip_max_norm,
            )
            # if batch_idx % 10 == 0:
            #     print(f"Training one batch took {time.time() - start}s")

            batch_idx += 1
        else:
            perform_activity(
                activity,
                power_consumption,
                paseos_instance,
                time_for_comms,
                constraint_function,
            )
            time_since_last_update += time_per_batch
            print(
                f"Rank {rank} standing by - Temperature[C]: "
                + f"{local_actor.temperature_in_K - 273.15:.2f},"
                + f"Battery SoC: {local_actor.state_of_charge:.2f}"
            )

        if plot and batch_idx % 10 == 0 and rank == 0:
            plotter.update(paseos_instance)

    paseos_instance.save_status_log_csv("results/" + str(rank) + ".csv")
    create_plots(paseos_instances=[paseos_instance], rank=rank)
    np.savetxt(
        "results/loss_rank" + str(rank) + ".csv", np.array(test_losses), delimiter=","
    )
    np.savetxt(
        "results/time_at_loss_rank" + str(rank) + ".csv",
        np.array(local_time_at_test),
        delimiter=",",
    )
    print(f"Rank {rank} finished.")


if __name__ == "__main__":
    print("Starting...")
    main(sys.argv[1:])
