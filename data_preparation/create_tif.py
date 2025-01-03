import os
import sys

sys.path.insert(1, "PyRawS")
from pyraws.raw.raw_event import Raw_event
import argparse
import torch
from termcolor import colored
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the THRAWS/raw directory.",
        default="PyRawS/data/THRAWS/raw",
    )

    parser.add_argument(
        "--bands",
        type=str,
        help='bands to coregister list in format ""[Bxx,Byy,...,Bzz]"".',
        default="[B02,B08,B03,B10,B04,B05,B11,B06,B07,B8A,B12,B01,B09]",
    )
    parser.add_argument(
        "--output_tif_dir",
        type=str,
        help="output TIF dir.",
        default="my_tif_dir",
    )

    pargs = parser.parse_args()
    requested_bands_str = pargs.bands
    requested_bands_str = requested_bands_str.replace(" ", "")[1:-1]
    bands = [x for x in requested_bands_str.split(",")]
    output_tif_dir = pargs.output_tif_dir
    input_dir = pargs.input_dir

    os.makedirs(output_tif_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # events_list = get_events_list("THRAWS")
    thraws_files = sorted(glob(os.path.join(input_dir, "*")))

    # events lift
    events_list = [file.split(os.sep)[-1] for file in thraws_files]

    for event, file in tqdm(zip(events_list, thraws_files), "Accessing event..."):
        print("Processing event: ", colored(event, "blue") + ".")
        try:
            raw_event = Raw_event(device=device)
            raw_event.from_path(file, bands, verbose=False)
        except:  # noqa E722
            print("Skipping event: ", colored(event, "red") + ".")
            continue

        if raw_event.is_void():
            print("Skipping event: ", colored(event, "red") + ".")
            continue

        raw_event_swir = Raw_event()
        raw_event_swir.from_path(file, ["B8A", "B11", "B12"], verbose=False)
        raw_event_rgb = Raw_event()
        raw_event_rgb.from_path(file, ["B02", "B03", "B04"], verbose=False)
        granules_list = list(range(len(raw_event.get_granules_info().keys())))

        for granule in granules_list:
            raw_granule_n = raw_event.get_granule(granule)
            save_path_n = os.path.join(pargs.output_tif_dir, event + "_" + str(granule))

            os.makedirs(save_path_n, exist_ok=True)
            print(
                "Exporting to tif file: " + colored(event + "_" + str(granule), "green")
            )
            raw_granule_n.export_to_tif(save_path_n)
            raw_granule_rgb_n = raw_event_rgb.coarse_coregistration([granule])
            raw_granule_swir_n = raw_event_swir.coarse_coregistration([granule])
            raw_granule_rgb_n.show_bands_superimposition()
            plt.savefig(
                os.path.join(save_path_n, event + "_" + str(granule) + "_rgb.png")
            )
            plt.close()
            raw_granule_swir_n.show_bands_superimposition()
            plt.savefig(
                os.path.join(save_path_n, event + "_" + str(granule) + "_swir.png")
            )

    print("processing " + colored("finished", "green") + ".")


if __name__ == "__main__":
    main()
