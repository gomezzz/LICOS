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


    for seed_target in tqdm([2], desc="Parsing seed..."):
        for qual_target in tqdm([1,2, 4,8], desc="Parsing qual..."):
            # Skip first combination


            for checkpoint in checkpoints:
                seed=int(checkpoint.split("seed=")[1].split("_")[0])
                qual = int(checkpoint.split("qual=")[1].split("_")[0])
                data_type = str(checkpoint.split("raw=")[1].split("_")[0])
                if seed == seed_target and qual == qual_target:
                    if data_type == "merged":
                        merged_path = os.path.join(checkpoints_path, checkpoint)
                        break

            cfg_path = "cfg"
            cfg_merged_name = "raw_merged_seed_"+str(seed_target)+"_q_"+str(qual_target)+".toml"
            cfg_merged = DotMap(toml.load(os.path.join(cfg_path, cfg_merged_name)), _dynamic=False)



            # Getting dataset path from cfg.
            dataset = cfg_merged.dataset

            print(f"Load model: {merged_path}.")
            # Selecting device
            device = "cuda" if torch.cuda.is_available() else "cpu"

            checkpoint_merged_path = (merged_path)
            net_merged = get_model(cfg_merged.model, False, 13, cfg_merged.model_quality)
            print(
                f"Merged Model has Parameters: {sum(p.numel() for p in net_merged.parameters())}"
            )
            checkpoint = torch.load(checkpoint_merged_path, map_location=device)
            net_merged.load_state_dict(checkpoint["state_dict"])
            net_merged.update()

            print("Loading merged dataset...")
            test_data_merged = RawImageFolder(
                    dataset,
                    cfg_merged.seed,
                    cfg_merged.raw_test_over_tot,
                    cfg_merged.raw_validation_over_train,
                    "merged",
                    cfg_merged.raw_target_resolution_merged_m,
                    split="test",
                    geographical_split_tolerance=cfg_merged.raw_train_test_tolerance,
                )

            results_df = pd.DataFrame(
            columns=["Type", "Seed", "PSNR", "SSIM", "BPP", "BPP_Pablo", "Compression"], index=[0]
            )

            print("Computing metrics")

            psnr_merged, ssim_merged, bpp_merged, bpp_pablo_merged, cr_merged= [], [], [], [], []

            compression_ratio = {}


            print("Running merged dataset")
            # Merged channels

            for img in tqdm(test_data_merged, desc="Processing dataset..."):
                # LICOS 13C
                img_size_in_bytes = img.shape[0] * img.shape[1] * img.shape[2]
                #print("Original merged image size: ", img_size_in_bytes, " bytes")
                img_uint8 = img_as_ubyte(img.squeeze(0).detach().cpu().numpy())
                out, _, _, compressed_size_in_bytes = process_img(
                    img, net_merged
                )
                img_reconstructed = img_as_ubyte(out["x_hat"].squeeze(0).squeeze(0).detach().cpu().numpy())
                psnr_merged.append(psnr(img_uint8, img_reconstructed))
                ssim_merged.append(ssim(img_uint8, img_reconstructed))
                bpp_pablo_merged.append(compute_bpp(out))
                bpp_merged.append(compressed_size_in_bytes/ img_size_in_bytes * 8 )
                cr_merged.append(compressed_size_in_bytes / img_size_in_bytes)


            print("MERGED", np.mean(psnr_merged), np.mean(ssim_merged), np.mean(bpp_merged), np.mean(bpp_pablo_merged))
            # Update results.
            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        {
                            "Type": "MERGED",
                            "Seed": cfg_merged.seed,
                            "PSNR": np.mean(psnr_merged),
                            "SSIM": np.mean(ssim_merged),
                            "BPP": np.mean(bpp_merged),
                            "BPP_Pablo" :  np.mean(bpp_pablo_merged),
                            "Compression": np.mean(cr_merged),
                        },
                        index=[0],
                    ),
                ],
            )

            output_name = "results_merged_seed_"+str(seed_target) + "_qual_"+ str(qual_target)+".csv"
            # Compute mean over different seeds
            results_df = results_df.groupby(["Type"]).mean().reset_index()
            results_df.to_csv(output_name, index=False)

            print(f"Saving output file: {output_name}.")


if __name__ == "__main__":
    main()