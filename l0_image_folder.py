import os
import rasterio
from torch.utils.data import Dataset
import numpy as np

from compressai.registry import register_dataset
import random
from glob import glob
from torchvision import transforms
from l0_utils import BAND_LIST, DN_MAX, image_band_reshape, IMAGE_SHAPE_DICT
import torch
from sklearn.model_selection import train_test_split


@register_dataset("L0ImageFolder")
class L0ImageFolder(Dataset):
    """L0 images data loader.

    Args:
        Dataset (Dataset): dataset class.
    """

    def __init__(
        self,
        root,
        seed,
        test_train_split,
        l0_format,
        target_resolution_merged_m=20.0,
        transform=None,
        preloaded=True,
        split="train",
    ):
        """Init function for L0ImageFolder.

        Args:
            root (string): root directory of the dataset
            seed (int): split seed.
            test_train_split (float): split percentage (e.g., 0.8 means 80% train and 20% test).
            l0_format (string): use "raw" to load bands separately, "merged" to load all the bands in one, "merged_with_res", to merge bands with the same resolution.
            target_resolution_merged_m (float, optional): target resolution in m when merged format is used. Defaults to 20.0.
            transform (callable, optional): a function or transform that takes in tensor and returns a transformed version.
            preloaded (bool, optional): if True, images are preloaded. Defaults to True.
            split (str, optional): split mode ('train' or 'test'). Defaults to "train".

        Raises:
            RuntimeError: Invalid directory.
            ValueError: Not implemented.
        """

        l0_files = sorted(glob(os.path.join(root, "*")))

        if (l0_files is None) or (len(l0_files) == 0):
            raise RuntimeError(f'Invalid directory "{root}"')

        # Splitting according to seed
        train_samples, test_samples, _, _ = train_test_split(
            l0_files,
            l0_files,  # Added since labels are needed but not used.
            test_size=(1 - test_train_split),
            random_state=seed,
        )

        if split == "train":
            self.samples = train_samples
        else:
            self.samples = test_samples

        # Target resolution in m for merged format.
        self.target_resolution_merged_m = target_resolution_merged_m

        # Set transform. Remove ToTensor() if included.
        if transform:
            self.transform = transforms.Compose(
                [
                    t
                    for t in transform.transforms
                    if not isinstance(t, transforms.ToTensor)
                ]
            )
        else:
            self.transform = None

        self.l0_format = l0_format

        # If True,images are preloaded
        self.preloaded = preloaded

        if self.l0_format == "raw":
            # Reshape samples to have a new sample for each band
            self._get_raw_format_()
            if preloaded:
                files = []
                for sample in self.samples:
                    files.append(self._open_band_(sample))

                # Samples are preloaded
                self.samples = files

        elif self.l0_format == "merged":
            if preloaded:
                files = []
                for sample in self.samples:
                    files.append(self._get_merged_file_(sample))
                # Samples are preloaded
                self.samples = files

        else:
            raise ValueError(self.l0_format + ".Not implemented.")

    def _get_raw_format_(self):
        """Reshape the samples list to have a new file for each band."""
        new_samples_list = []

        for sample in self.samples:
            for n in range(13):
                new_samples_list.append(os.path.join(sample, BAND_LIST[n] + ".tif"))

        self.samples = new_samples_list

    def _get_merged_file_(self, file_path):
        """
        Merge all the bands for a single file in a single l0_file by upsample.
        Args:
            file_path (str): path to the band file.
        Returns:
            img (torch): output tensor.
        """

        # Creating a placeholder for the reshaped bands
        img = torch.zeros(
            13,
            IMAGE_SHAPE_DICT[self.target_resolution_merged_m][0],
            IMAGE_SHAPE_DICT[self.target_resolution_merged_m][1],
        )

        for n in range(0, 12):
            # Reshaping all the bands to the target resolution
            img[n] = image_band_reshape(
                self._open_band_(
                    os.path.join(file_path, BAND_LIST[n] + ".tif")
                ).squeeze(0),
                BAND_LIST[n],
                self.target_resolution_merged_m,
            )

        return img

    def _open_band_(self, band_path):
        """
        Args:
            band_path (str): path to the band file.
        Returns:
            band (torch): output tensor.
        """

        with rasterio.open(band_path) as src:
            return (
                torch.from_numpy(src.read(1).astype(np.float32)).unsqueeze(0) / DN_MAX
            )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        if self.preloaded:
            img = self.samples[index]
        else:
            if self.l0_format == "raw":
                img = self._open_band_(self.samples[index])
            elif self.l0_format == "merged":
                img = self._get_merged_file_(self.samples[index])

        if self.transform:
            return self.transform(img)
        else:
            return img

    def __len__(self):
        return len(self.samples)