import sys
import time

import paseos

from utils import (
    parse_args,
    save_checkpoint,
)
from create_plots import create_plots
from init_paseos import init_paseos
from train import train_one_batch, train_one_epoch, test_epoch, init_training


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
    if paseos_instance.local_actor.temperature_in_K > (273.15 + 45):
        return False
    if paseos_instance.local_actor.state_of_charge < 0.2:
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
        or paseos_instance.local_actor.state_of_charge < 0.25
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


def main(argv):
    # Init
    rank = 0  # compute index of this node
    test_freq = 500  # after how many it to eval test set
    time_per_batch = 0.1 * 100  # estimated time per batch in seconds
    time_in_standby = 0
    standby_period = 600  # how long to standby if necessary
    plot = False
    constraint_function = lambda: constraint_func(paseos_instance, groundstations)
    args = parse_args(argv)
    paseos.set_log_level("INFO")
    args.pretrained = False

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
    ) = init_training(args)

    # Init paseos
    paseos_instance, local_actor, groundstations = init_paseos(rank)

    if plot:
        plotter = paseos.plot(paseos_instance, paseos.PlotType.SpacePlot)

    # Training loop
    best_loss = float("inf")
    batch_idx = 0
    while batch_idx < args.epochs:
        if batch_idx % 100 == 0:
            print(
                f"Rank {rank} - Temperature[C]: {local_actor.temperature_in_K - 273.15:.2f}, Battery SoC: {local_actor.state_of_charge:.2f}"
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
            # print(f"Training one batch took {time.time() - start}s")

            if batch_idx % test_freq == 0:
                print(f"Evaluating test set")
                print(f"Previous learning rate: {optimizer.param_groups[0]['lr']}")
                start = time.time()
                loss = test_epoch(batch_idx, test_dataloader, net, criterion)
                print(f"Test evaluation took {time.time() - start}s")

                lr_scheduler.step(loss)
                print(f"New learning rate: {optimizer.param_groups[0]['lr']}")

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
                    )
            batch_idx += 1
        else:
            print(
                f"Rank {rank} standing by - Temperature[C]: {local_actor.temperature_in_K - 273.15:.2f}, Battery SoC: {local_actor.state_of_charge:.2f}"
            )

        if plot and batch_idx % 10 == 0:
            plotter.update(paseos_instance)

    paseos_instance.save_status_log_csv("results/" + str(rank) + ".csv")
    create_plots(paseos_instances=[paseos_instance])


if __name__ == "__main__":
    main(sys.argv[1:])
