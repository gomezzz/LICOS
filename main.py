import sys
import time
import os

import torch
import numpy as np
import pykep as pk
import sys

sys.path.insert(1, "PASEOS")
import paseos
from mpi4py import MPI

from utils import parse_args
from create_plots import create_plots
from init_paseos import init_paseos
from actor_logic import constraint_func, decide_on_activity, perform_activity
from federation_utils import update_central_model
from train import train_one_batch, init_training, eval_test_set


def main(argv):
    # Init
    rank = 0  # compute index of this node
    time_per_batch = 0.1 * 100  # estimated time per batch in seconds
    assert time_per_batch < 30, "For a high time per batch you may miss comms windows?"
    time_for_comms = 60
    time_in_standby = 0
    time_since_last_update = 0
    total_simulation_time = 0
    standby_period = 900  # how long to standby if necessary
    MPI_sync_period = 600  # After how many seconds we wait synchronize instance clocks

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
    time_of_last_sync = local_actor.local_time.mjd2000 * pk.DAY2SEC

    print(f"Rank {rank} - Init PASEOS")
    sys.stdout.flush()

    print(f"Rank {rank} - Init PASEOS")
    sys.stdout.flush()

    if plot and rank == 0:
        plotter = paseos.plot(paseos_instance, paseos.PlotType.SpacePlot)

    ################################################################################
    # Main Training loop
    best_loss = float("inf")
    batch_idx = 0
    while total_simulation_time < args.simulation_time:

        ################################################################################
        # Sync time between ranks to minimize divergence
        if (
            local_actor.local_time.mjd2000 * pk.DAY2SEC - time_of_last_sync
        ) > MPI_sync_period:
            print(f"Rank {rank} waiting for sync", end=" ")
            sys.stdout.flush()
            comm.Barrier()
            time_of_last_sync = local_actor.local_time.mjd2000 * pk.DAY2SEC
            if rank == 0:
                print()

        if batch_idx % 100 == 0:
            print(
                f"Rank {rank} - {str(paseos_instance.local_actor.local_time)} - Temperature[C]: "
                + f"{local_actor.temperature_in_K - 273.15:.2f},"
                + f"Battery SoC: {local_actor.state_of_charge:.2f}"
            )
            sys.stdout.flush()
            # print(f"PASEOS advancing time by {time_per_batch}s.")

        ################################################################################
        # Decide what this rank will do in this time step
        activity, power_consumption, time_in_standby = decide_on_activity(
            paseos_instance,
            time_per_batch,
            time_for_comms,
            time_in_standby,
            standby_period,
            time_since_last_update,
        )

        ################################################################################
        # Perform  what was the decided on first in paseos and than on the rank, either
        # A) Exchange model with ground
        # B) Traing model on a batch
        # C) Standby to cool down / recharge
        if activity == "Model_update":
            print(
                f"Rank {rank} will update with GS "
                + str(list(paseos_instance.known_actors.items())[0][0])
                + " at "
                + str(paseos_instance.local_actor.local_time)
            )
            # 1) Model comms in PASEOS (already know there is a window from decide on activity)
            perform_activity(
                activity,
                power_consumption,
                paseos_instance,
                time_for_comms,
                constraint_function,
            )
            # 2) Evaluate test set before exchanging models
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

            # 3) Exchange models with the ground stations
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

            # 4) Evaluate test set after exchanging models
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
            # 1) Model training cost in PASEOS
            perform_activity(
                activity,
                power_consumption,
                paseos_instance,
                time_per_batch,
                constraint_function,
            )
            time_since_last_update += time_per_batch
            # 2) Train model on one batch
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
            # 1) Model standby in PASEOS and do nothing :)
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

        total_simulation_time = (
            paseos_instance._state.time - paseos_instance._cfg.sim.start_time
        )

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
