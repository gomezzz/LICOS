import sys
import time

import paseos

from utils import (
    parse_args,
    save_checkpoint,
)
from create_plots import create_plots
from init_paseos import init_paseos
from train import train_one_epoch, test_epoch, init_training


def main(argv):
    # Init
    rank = 0
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
    ) = init_training(args)

    # Init paseos
    paseos_instance, local_actor = init_paseos(rank)

    plotter = paseos.plot(paseos_instance, paseos.PlotType.SpacePlot)

    # Training loop
    best_loss = float("inf")
    time_per_epoch = 112 * 5
    for epoch in range(last_epoch, args.epochs):
        print(
            f"Rank {rank} - Temperature: {local_actor.temperature_in_K - 273.15}, Battery: {local_actor.state_of_charge}, In_Eclpise: {local_actor.is_in_eclipse()}"
        )
        print(f"PASEOS advancing time by {time_per_epoch}s.")

        # Wattage from 1605B https://www.amd.com/en/products/embedded-ryzen-v1000-series
        # https://unibap.com/wp-content/uploads/2021/06/spacecloud-ix5-100-product-overview_v23.pdf
        # 1400 base consumption for S2 from https://sentinels.copernicus.eu/documents/247904/349490/S2_SP-1322_2.pdf
        paseos_instance.advance_time(
            time_per_epoch, current_power_consumption_in_W=1400 + 30
        )

        # Train one
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        start = time.time()
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        print(f"Training one epoch took {time.time() - start}s")

        start = time.time()
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        print(f"Test evaluation took {time.time() - start}s")

        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
            )

        plotter.update(paseos_instance)

    paseos_instance.save_status_log_csv("results/" + str(rank) + ".csv")
    create_plots(paseos_instances=[paseos_instance])


if __name__ == "__main__":
    main(sys.argv[1:])
