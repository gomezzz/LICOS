import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
from PIL import Image
import warnings
# Ignore pandas future warnigns
warnings.simplefilter(action="ignore", category=FutureWarning)

from dotmap import DotMap
import numpy as np
import toml
import torch
from glob import glob
from tqdm import tqdm
import pandas as pd

sys.path.append("./licos/")
from licos.model_utils import get_model
from licos.raw_image_folder import RawImageFolder

from eval_utils import (
    compute_bpp,
    compute_msssim,
    compute_psnr,
    find_closest_bpp,
    process_img,
)
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.util import img_as_ubyte

def main():
    # checkpoints
    checkpoints_path = os.path.join("licos", "results")
    checkpoints = [x.split(os.sep)[-1] for x in sorted(glob(os.path.join(checkpoints_path, "*"))) if not(os.path.isdir(x))]
    # Removing unwanted files
    checkpoints = checkpoints[:-5]

    # Extracting checkpoints
    checkpoints_path = os.path.join("licos", "results")
    checkpoints = [x.split(os.sep)[-1] for x in sorted(glob(os.path.join(checkpoints_path, "*"))) if not(os.path.isdir(x))]
    # Removing unwanted files
    checkpoints = checkpoints[:-5]

    # Extract baseline models flag to avoid to do it more than once
    baseline_extracted = False

    for seed_target in tqdm([2], desc="Parsing seed..."):
        for qual_target in tqdm([8], desc="Parsing qual..."):

            for checkpoint in checkpoints:
                seed=int(checkpoint.split("seed=")[1].split("_")[0])
                qual = int(checkpoint.split("qual=")[1].split("_")[0])
                data_type = str(checkpoint.split("raw=")[1].split("_")[0])
                if seed == seed_target and qual == qual_target:
                    if data_type == "split":
                        split_path = os.path.join(checkpoints_path, checkpoint)
                        break

            cfg_path = "cfg"
            cfg_split_name = "raw_split_seed_"+str(seed_target)+"_q_"+str(qual_target)+".toml"
            cfg_split = DotMap(toml.load(os.path.join(cfg_path, cfg_split_name)), _dynamic=False)



            # Getting dataset path from cfg.
            dataset = cfg_split.dataset

            print(f"Load model: {split_path}.")
            # Selecting device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            split_checkpoint_path = (split_path)
            print("Loading models at {}".format(split_checkpoint_path))
            net_split = get_model(cfg_split.model, False, 1, cfg_split.model_quality)
            print(
                f"Split Models has Parameters: {sum(p.numel() for p in net_split.parameters())}"
            )
            checkpoint = torch.load(split_checkpoint_path, map_location=device)
            net_split.load_state_dict(checkpoint["state_dict"])
            net_split.update()

            print("Loading split dataset...")
            test_data_split = RawImageFolder(
                dataset,
                cfg_split.seed,
                cfg_split.raw_test_over_tot,
                cfg_split.raw_validation_over_train,
                "split",
                cfg_split.raw_target_resolution_merged_m,
                split="test",
                geographical_split_tolerance=cfg_split.raw_train_test_tolerance,
            )

            results_df = pd.DataFrame(
            columns=["Type", "Seed", "PSNR", "SSIM", "BPP", "BPP_Pablo", "Compression"], index=[0]
            )

            print("Computing metrics")

            psnr_split, ssim_split, bpp_split, bpp_pablo_split, cr_split = [], [], [], [], []



            print("Running split dataset")
            # Split channels

            for img in tqdm(test_data_split, desc="Processing dataset..."):
                img_size_in_bytes = img.shape[0] * img.shape[1] * img.shape[2]
                print("Original split image size: ", img_size_in_bytes, " bytes")
                img_uint8 = img_as_ubyte(img.squeeze(0).detach().cpu().numpy())
                out, _, _, compressed_size_in_bytes = process_img(img, net_split)
                img_reconstructed = img_as_ubyte(out["x_hat"].squeeze(0).squeeze(0).detach().cpu().numpy())
                psnr_split.append(psnr(img_uint8, img_reconstructed))
                ssim_split.append(ssim(img_uint8, img_reconstructed))
                bpp_pablo_split.append(compute_bpp(out))
                bpp_split.append(compressed_size_in_bytes/ img_size_in_bytes * 8 )
                cr_split.append(compressed_size_in_bytes / img_size_in_bytes)


            print("SPLIT", np.mean(psnr_split), np.mean(ssim_split), np.mean(bpp_split), np.mean(bpp_pablo_split))
            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        {
                            "Type": "SPLIT",
                            "Seed": cfg_split.seed,
                            "PSNR": np.mean(psnr_split),
                            "SSIM": np.mean(ssim_split),
                            "BPP": np.mean(bpp_split),
                            "BPP_Pablo" :  np.mean(bpp_pablo_split),
                            "Compression": np.mean(cr_split),
                        },
                        index=[0],
                    ),
                ]
            )



            output_name = "results_split_seed_"+str(seed_target) + "_qual_"+ str(qual_target)+".csv"
            # Compute mean over different seeds
            results_df = results_df.groupby(["Type"]).mean().reset_index()
            results_df.to_csv(output_name, index=False)

            print(f"Saving output file: {output_name}.")


if __name__ == "__main__":
    main()