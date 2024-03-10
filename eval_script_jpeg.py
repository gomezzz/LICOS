import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import sys
from PIL import Image
import warnings
# Ignore pandas future warnigns
warnings.simplefilter(action="ignore", category=FutureWarning)

from dotmap import DotMap
import numpy as np
import toml
from torchvision import transforms
from glob import glob
from tqdm import tqdm
import pandas as pd

sys.path.append("./licos/")
from PIL import Image
from licos.raw_image_folder import RawImageFolder
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.util import img_as_ubyte

from eval_utils import (
    compute_msssim,
    compute_psnr,
    pillow_encode,
)

out_dir="/home/gabrielemeoni/project/LICOS/JPEG_out_files"

def main():

    transform = transforms.Compose([transforms.PILToTensor()])

    qual_target = 1
    for seed_target in tqdm([2], desc="Parsing seed..."):

        cfg_path = "cfg"
        cfg_split_name = "raw_split_seed_"+str(seed_target)+"_q_"+str(qual_target)+".toml"
        cfg_split = DotMap(toml.load(os.path.join(cfg_path, cfg_split_name)), _dynamic=False)


        # Getting dataset path from cfg.
        dataset = cfg_split.dataset


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
        columns=["Type", "Seed", "PSNR", "SSIM", "BPP", "Compression"], index=[0]
        )

        print("Computing metrics")

        cr_list = []
        psnr_list = []
        ssim_list = []
        bpp_list = []
        q_list = []

        for q in [1, 2, 3, 4, 5, 6, 10]:
            desc = "Parsing seed: " + str(seed_target) + ", q: " + str(q)+"..."
            psnr_jpeg = []
            ssim_jpeg = []
            cr_jpeg = []
            bpp_jpeg = []
            n_image = 0
            for img in tqdm(test_data_split, desc=desc):
                img_uint8 = img_as_ubyte(img.squeeze(0))
                PIL_img = Image.fromarray(img_uint8, "L")
                img_jpeg, bpp_jpeg_img = pillow_encode(PIL_img, fmt="jpeg", quality=int(q))
                img_jpeg.save(os.path.join(out_dir, "jpeg_n_img_"+str(n_image) + "_q_"+str(q)+".jpeg"))
                img_jpeg=transform(img_jpeg).detach().cpu().numpy().squeeze(0)

                psnr_jpeg.append(psnr(img_uint8, img_jpeg))
                ssim_jpeg.append(ssim(img_uint8, img_jpeg))
                cr_jpeg.append(bpp_jpeg_img  / 8)
                bpp_jpeg.append(bpp_jpeg_img)
                n_image +=1

            q_list.append(q)
            ssim_list.append(np.mean(ssim_jpeg))
            psnr_list.append(np.mean(psnr_jpeg))
            cr_list.append(np.mean(cr_jpeg))
            bpp_list.append(np.mean(bpp_jpeg))


        print("Running split dataset")
        # Split channels


        for q in range(len(q_list)):
            if q == 0:
                results_df = pd.DataFrame(
                    {
                        "Type": "SPLIT",
                        "Seed": cfg_split.seed,
                        "PSNR": psnr_list[q],
                        "SSIM": ssim_list[q],
                        "BPP": bpp_list[q],
                        "q" : q_list[q],
                        "Compression": cr_list[q],
                    },
                    index=[0],
                )
            else:
                new_df = pd.DataFrame(
                    {
                        "Type": "SPLIT",
                        "Seed": cfg_split.seed,
                        "PSNR": psnr_list[q],
                        "SSIM": ssim_list[q],
                        "BPP": bpp_list[q],
                        "q" : q_list[q],
                        "Compression": cr_list[q],
                    },
                    index=[0],
                )
                results_df = pd.concat([results_df, new_df])

        output_name = "results_jpeg_seed_"+str(seed_target) +".csv"
        # Compute mean over different seeds

        results_df.to_csv(output_name, index=False)

        print(f"Saving output file: {output_name}.")


if __name__ == "__main__":
    main()