from torch.nn.functional import interpolate
import torch

from copy import deepcopy
import random
import numpy as np
import os


def geographical_splitter(
    filenames, test_size_percentage, seed=42, split_percentage_error_tolerance=0.01
):
    """Splits the events according to a geographical position. In this way, patches related to a specific area can be only in train or in test.

    Args:
        filenames (list): file names of the different images.
        test_size_percentage (float): split perecentage.
        seed (int, optional): seed for reproducibility. Defaults to 42.
        split_percentage_error_tolerance (float, optional): tolerance on the split percentage error. Defaults to 0.01.

    Returns:
        list: train filenames
        list: test filenames
    """
    # locations - n_files dictionary
    location_files_dictionary = {
        "Fuego": 16,
        "Sangay": 35,
        "Piton_de_la_Fournaise": 31,
        "Chillan_Nevados_de": 6,
        "Barren_Island": 20,
        "Nyamulagira": 18,
        "Copahue": 6,
        "Krysuvik-Trolladyngja": 24,
        "Santa_Maria": 3,
        "San_Miguel": 14,
        "Mayon": 6,
        "Stromboli": 10,
        "Raung": 15,
        "Etna": 14,
        "La_Palma": 10,
        "Karangetang": 6,
        "Telica": 6,
        "Tinakula": 9,
        "Bolivia": 9,
        "Sweden": 5,
        "Kenya": 6,
        "Greece": 31,
        "Mexico": 6,
        "Greenland": 7,
        "Ukraine": 6,
        "Italy": 7,
        "Latvia": 16,
        "Australia": 3,
        "Spain": 8,
        "France": 5,
    }

    # Copying locations to perform shuffle
    location_files_dictionary_shuffled = deepcopy(
        list(location_files_dictionary.keys())
    )

    # Fixing seed to ensure that dataset train and test will be splitted in the same way into different iterations
    random.seed(seed)
    # Shuffling locations
    random.shuffle(location_files_dictionary_shuffled)

    # Number of total files
    n_files = np.sum(np.array([n for n in location_files_dictionary.values()]))

    # Maximum number of events in tests
    n_files_test_max = int(test_size_percentage * n_files)

    # Number of files in tests
    n_files_test = 0

    # Currernt index
    idx = 0

    # List of files location in test
    files_locations_tests_list = []

    while n_files_test < n_files_test_max:
        n_files_test += location_files_dictionary[
            location_files_dictionary_shuffled[idx]
        ]
        files_locations_tests_list.append(location_files_dictionary_shuffled[idx])
        idx += 1

    real_percentage = (n_files_test) / n_files

    if real_percentage - test_size_percentage > split_percentage_error_tolerance:
        raise ValueError(
            "Impossible to perform datatest TRAIN/EVAL "
            + str(test_size_percentage)
            + " splitting with tolerance: "
            + str(split_percentage_error_tolerance * 100)
            + " % by using SEED: "
            + str(seed)
            + ". Try to change seed."
        )

    # Placheholders
    test_files = []
    train_files = []

    # Splitting files according to the selected locations
    for filename in filenames:
        filename_tr = filename.split(os.sep)[-1]
        if filename_tr[-4] == "_":
            # Fire event (Format: location_idx_granule_number)
            location = filename_tr[:-4]
        else:
            # Volcano event (Format: location_idx0idx1_granule_number)
            location = filename_tr[:-5]
        # If the location of the file is in the designated test ones, move the file into test files.
        if location in files_locations_tests_list:
            test_files.append(filename)
        else:
            train_files.append(filename)
    return train_files, test_files


# Sentinel-2 band names
BAND_LIST = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B09",
    "B8A",
    "B10",
    "B11",
    "B12",
]

# Spatial resolution of various bands
BAND_SPATIAL_RESOLUTION_DICT = dict(
    zip(BAND_LIST, [60, 10, 10, 10, 20, 20, 20, 10, 60, 20, 60, 20, 20])
)

# Digital number max - 12 bits
DN_MAX = 2**12 - 1

# Target image shape
IMAGE_SHAPE_DICT = {10.0: [2304, 2592], 20.0: [1152, 1296], 60.0: [384, 432]}


def image_band_upsample(img_band, band_name, upsample_factor, upsample_mode="bilinear"):
    """Upsample an image band to a target spatial resolution through an upsample mode.

    Args:
        img_band (torch.tensor): image band.
        band_name (string): band name.
        upsample_factor (int): upsample factor.
        upsample_mode (string, optional): "nearest", "bilinear", "bicubic". Defaults to blinear.

    Raises:
        ValueError: unsupported band name.
        ValueError: unsupported upsample mode.

    Returns:
    """

    if not (upsample_mode in ["nearest", "bilinear", "bicubic"]):
        raise ValueError(
            "Upsample mode " + upsample_mode + " not supported. Please, choose among: "
            "nearest"
            ", "
            "bilinear"
            ", "
            "bicubic"
            "."
        )

    if BAND_SPATIAL_RESOLUTION_DICT[band_name] == 60:
        # Using different upsampling factors since 60 m bands
        # have 60 m resolution vertically, but 20 m resolution horizontally.
        upsample_factor = (upsample_factor, upsample_factor / 3)

    with torch.no_grad():
        if upsample_mode == "nearest":
            return (
                interpolate(
                    img_band.unsqueeze(0).unsqueeze(0),
                    scale_factor=upsample_factor,
                    mode=upsample_mode,
                )
                .squeeze(0)
                .squeeze(0)
            )
        else:
            return (
                interpolate(
                    img_band.unsqueeze(0).unsqueeze(0),
                    scale_factor=upsample_factor,
                    mode=upsample_mode,
                    align_corners=True,
                )
                .squeeze(0)
                .squeeze(0)
            )


def image_band_reshape(
    img_band,
    band_name,
    target_resolution,
    upsample_mode="bilinear",
    downsample_mode="bilinear",
):
    """Reshape a band to a target resolution.

    Args:
        img_band (torch.tensor): image band.
        band_name (str): band name.
        target_resolution (float): target resolution in m.
        upsample_mode (str, optional): "nearest", "bilinear", "bicubic". Defaults to blinear.
        downsample_mode (str, optional): "nearest", "bilinear", "bicubic". If None, pixels are just dicarded. Defaults to blinear.
    Raises:
        ValueError: Unsupported band name

    Returns:
        torch.tensor: resampled band.
    """
    if not (band_name in BAND_LIST):
        raise ValueError("Unsupported band name: " + band_name + ".")

    # Calculating upsample factor
    upsample_factor = BAND_SPATIAL_RESOLUTION_DICT[band_name] / target_resolution

    if upsample_factor > 1:
        return image_band_upsample(
            img_band, band_name, int(upsample_factor), upsample_mode=upsample_mode
        )

    elif upsample_factor < 1:
        if downsample_mode is None:
            downsample_factor = int(1 / upsample_factor)
            return img_band[::downsample_factor, ::downsample_factor]
        else:
            return (
                interpolate(
                    img_band.unsqueeze(0).unsqueeze(0),
                    scale_factor=upsample_factor,
                    mode=downsample_mode,
                )
                .squeeze(0)
                .squeeze(0)
            )

    else:
        if BAND_SPATIAL_RESOLUTION_DICT[band_name] == 60:
            # Downsampling image across-track to 60 m.
            # 60m-bands have 60 m resolution vertically but 20 m resolution horizontally.
            return img_band[::, ::3]
        else:
            return img_band
