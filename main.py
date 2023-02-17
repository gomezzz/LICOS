import sys
import time

import torch
import numpy as np
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
    paseos_instance: paseos.PASEOS, timestep, time_in_standby, standby_period
):
    """Heuristic to decide activitiy for the actor. Initiates a standby period of passed
    length when going into standby.

    Args:
        paseos_instance (paseos): Local instance

    Returns:
        activity,power_consumption
    """
    if (
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


def aggregate(rank_model, received_models, device):
    cw = 1 / (len(received_models) + 1)

    local_sd = rank_model.state_dict()
    for key in local_sd:
        local_sd[key] = cw * local_sd[key].to(device) + sum(
            [sd[key].to(device) * cw for i, sd in enumerate(received_models)]
        )

    # update server model with aggregated models
    rank_model.load_state_dict(local_sd)


def main(argv):
    # Init
    rank = 0  # compute index of this node
    test_freq = 1000  # after how many it to eval test set
    time_per_batch = 0.1 * 100  # estimated time per batch in seconds
    time_in_standby = 0
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

    # Init paseos
    paseos_instance, local_actor, groundstations = init_paseos(rank, comm.Get_size())

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
            # print(f"PASEOS advancing time by {time_per_batch}s.")

        activity, power_consumption, time_in_standby = decide_on_activity(
            paseos_instance, time_per_batch, time_in_standby, standby_period
        )
        perform_activity(
            activity,
            power_consumption,
            paseos_instance,
            time_per_batch,
            constraint_function,
        )

        if activity == "Training":
            # Train one
            start = time.time()
            train_dataloader_iter = train_one_batch(
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

            if batch_idx % test_freq == 0:
                print(f"Rank {rank} - Evaluating test set")
                print(
                    f"Rank {rank} - Previous learning rate: {optimizer.param_groups[0]['lr']}"
                )
                start = time.time()
                loss = test_epoch(batch_idx, test_dataloader, net, criterion)
                test_losses.append(loss.item())
                local_time_at_test.append(
                    paseos_instance.local_actor.local_time.mjd2000()
                )
                print(f"Rank {rank} - Test evaluation took {time.time() - start}s")

                lr_scheduler.step(loss)
                print(
                    f"Rank {rank} - New learning rate: {optimizer.param_groups[0]['lr']}"
                )

                is_best = loss < best_loss
                best_loss = min(loss, best_loss)

                if args.save:
                    save_checkpoint(
                        {
                            "batch_idx": batch_idx,
                            "state_dict": net.state_dict(),
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                        },
                        is_best,
                        filename="checkpoint_rank" + str(rank) + ".pth.tar",
                    )

                # Wait for all ranks to evaluate the test set again
                comm.Barrier()
                print(f"Rank {rank} waiting to share models.")
                sys.stdout.flush()

                other_models = []
                # Load other models
                for other_rank in other_ranks:
                    state_dict = torch.load(
                        "checkpoint_rank" + str(other_rank) + ".pth.tar",
                        map_location=device,
                    )["state_dict"]
                    other_models.append(state_dict)

                # Aggregate
                aggregate(net, other_models, device)
                print(f"Rank {rank} finished aggregating models.")
                sys.stdout.flush()

            batch_idx += 1
        else:
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


if __name__ == "__main__":
    main(sys.argv[1:])
