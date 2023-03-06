import os
import rasterio
from torch.utils.data import Dataset
import numpy as np

from compressai.registry import register_dataset
import random
from glob import glob
from torchvision import transforms
from l0_utils import BAND_LIST, BAND_SPATIAL_RESOLUTION_DICT, image_band_upsample
import torch

@register_dataset("L0ImageFolder")
class L0ImageFolder(Dataset):
    """Load an image folder database. Splitting between training and test is done at runtime with a fixed seed.
    Args:
        root (string): root directory of the dataset
        seed (int): split seed. 
        test_train_split (float): split percentage (e.g., 0.8 means 80% train and 20% test).
        l0_format (string): use "raw" to load bands separately, "merged" to load all the bands in one, "merged_with_res", to merge bands with the same resolution.
        transform (callable, optional): a function or transform that takes in tensor and returns a transformed version.
        preloaded (bool): if True, images are preloaded. 
        split (string): split mode ('train' or 'test')
    """

    def __init__(self, root, seed, test_train_split,l0_format, transform=None, preloaded=True, split="train"):
        
        l0_files=sorted(glob(os.path.join(root, "*")))

        if (l0_files is None) or (len(l0_files) == 0):
            raise RuntimeError(f'Invalid directory "{root}"')
        
        random.seed(seed)
        
        random.shuffle(l0_files)
        
        train_split_idx=int(test_train_split * len(l0_files)) + 1
        
        if split == "train":
            self.samples=l0_files[:train_split_idx]
        else:
            self.samples=l0_files[train_split_idx:]

            
        #Set transform. Remove ToTensor() if included.
        if transform:
            self.transform = self.__clean_transform__(transform)#transforms.Compose([transforms.ToPILImage()]+ [x for x in transform.transforms])
        else:
            self.transform=None
        
        self.l0_format = l0_format
        
        #If True,images are preloaded
        self.preloaded=preloaded
        
        if self.l0_format == "raw":
            #Reshape samples to have a new sample for each band
            self.__get_raw_format__()
            
            if preloaded:
                files=[]
                for sample in self.samples:
                    files.append(self.__open_band__(sample))
                
                #Samples are preloaded
                self.samples=files
                
        elif (self.l0_format == "merged") and preloaded:
            files=[]
            for sample in self.samples:
                files.append(self.__get_merged_file__(sample))
            #Samples are preloaded
            self.samples=files
            
        else:
            raise ValueError(self.l0_format+".Not implemented.") 
    
    def __clean_transform__(self, transform):
        """Remove ToTensor() to avoid errors
        Args:
            transform (transform): torchvision transform
        Returns:
            transform (transform): cleaned transform
        """
        transform_list=[]

        for transform in transform.transforms:
            if str(transform) != "ToTensor()":
                transform_list.append(transform)

        if len(transform_list) != 1:
            return transforms.Compose(transform_list)
        else:
            return transform_list[0]
    
            
    def __get_raw_format__(self):
        """Reshape the samples list to have a new file for each band."""
        new_samples_list=[]
        
        for sample in self.samples:
            for n in range(13):
                new_samples_list.append(os.path.join(sample, BAND_LIST[n]+".tif"))
            
        self.samples=new_samples_list
        
    def __get_merged_file__(self, file_path):
        """
        Merge all the bands for a single file in a single l0_file by upsample.
        Args:
            file_path (str): path to the band file.
        Returns:
             (torch): output tensor.
        """
        band_2=self.__open_band__(os.path.join(file_path, BAND_LIST[2]+".tif"))
        
        band_tensor=torch.zeros(13, band_2.shape[0], band_2.shape[1])
        band_tensor[2]=band_2
        for n in [1] + [x for x in range(3,13)]:
            band_tensor[n]=image_band_upsample(self.__open_band__(os.path.join(file_path, BAND_LIST[n]+".tif")), BAND_LIST[n])
        
        return band_tensor
        
        
    def __open_band__(self, band_path):
        """
        Args:
            band_path (str): path to the band file.
        Returns:
            band (torch): output tensor.
        """
        
        with rasterio.open(band_path) as src:
            return torch.from_numpy(src.read(1).astype(np.float32)).unsqueeze(0)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        if self.preloaded:
            img=self.samples[index]
        else:
            if self.l0_format == "raw":
                img=self.__open_band__(self.samples[index])
            elif self.l0_format == "merged":
                img=self.__get_merged_file__(self.samples[index])
        
        if self.transform:
            return self.transform(img)
        else: 
            return img

    def __len__(self):
        return len(self.samples)