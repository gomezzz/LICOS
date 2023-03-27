import sys
from PIL import Image
import warnings

# Ignore pandas future warnigns
warnings.simplefilter(action="ignore", category=FutureWarning)

from dotmap import DotMap
import numpy as np
import toml
import torch
from tqdm import tqdm
import pandas as pd

sys.path.append("./licos/")

from licos.model_utils import get_model
from licos.l0_image_folder import L0ImageFolder

from eval_utils import (
    compute_bpp,
    compute_msssim,
    compute_psnr,
    find_closest_bpp,
    find_closest_psnr,
    find_closest_msssim,
    process_img,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
seeds = [2]
cfg = DotMap(toml.load("cfg/l0.toml"), _dynamic=False)

# Small dataset
dataset = "/home/pablo/raw_test/"
cfg.l0_train_test_tolerance = 0.5
cfg.l0_test_over_tot = 0.5
cfg.l0_validation_over_train = 0.5

# Large dataset
# dataset = "/home/pablo/rawdata/my_tif_dir"

results_df = pd.DataFrame(
    columns=["Type", "Seed", "PSNR", "SSIM", "BPP", "Compression"]
)

for seed in seeds:
    cfg.seed = seed
    checkpoint_path = (
        "results/bmshj2018-factorizedqual=1_l0=raw_seed=" + str(seed) + ".pth.tar"
    )
    checkpoint_merged_path = (
        "results/bmshj2018-factorizedqual=1_l0=merged_seed=" + str(seed) + ".pth.tar"
    )

    print("Loading models at {} and {}".format(checkpoint_path, checkpoint_merged_path))

    net_raw = get_model(cfg.model, False, 1, cfg.model_quality)
    print(f"Raw Models has Parameters: {sum(p.numel() for p in net_raw.parameters())}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net_raw.load_state_dict(checkpoint["state_dict"])
    net_raw.update()

    net_merged = get_model(cfg.model, False, 13, cfg.model_quality)
    print(
        f"Merged Model has Parameters: {sum(p.numel() for p in net_merged.parameters())}"
    )
    checkpoint = torch.load(checkpoint_merged_path, map_location=device)
    net_merged.load_state_dict(checkpoint["state_dict"])
    net_merged.update()

    print("Loading dataset...")
    test_data_raw = L0ImageFolder(
        dataset,
        cfg.seed,
        cfg.l0_test_over_tot,
        cfg.l0_validation_over_train,
        "raw",
        cfg.l0_target_resolution_merged_m,
        split="test",
        geographical_split_tolerance=cfg.l0_train_test_tolerance,
    )
    test_data_merged = L0ImageFolder(
        dataset,
        cfg.seed,
        cfg.l0_test_over_tot,
        cfg.l0_validation_over_train,
        "merged",
        cfg.l0_target_resolution_merged_m,
        split="test",
        geographical_split_tolerance=cfg.l0_train_test_tolerance,
    )

    print("Computing metrics")
    psnr_raw, ssim_raw, bpp_raw = [], [], []
    psnr_merged, ssim_merged, bpp_merged = [], [], []

    psnr_bbp_matched, ssim_bbp_matched, bpp_bbp_matched = [], [], []
    psnr_psnr_matched, ssim_psnr_matched, bpp_psnr_matched = [], [], []
    psnr_ssim_matched, ssim_ssim_matched, bpp_ssim_matched = [], [], []

    compression_ratio = {}

    print("Running merged dataset")
    # Merged channels
    for img in tqdm(test_data_merged):
        # LICOS 13C
        img_size_in_bytes = 8 * img.shape[0] * img.shape[1] * img.shape[2]
        print("Original merged image size: ", img_size_in_bytes, " bytes")
        out, reconstructed, diff, compressed_size_in_bytes = process_img(
            img, net_merged
        )
        out_bpp = compute_bpp(out)
        out_psnr = compute_psnr(img.unsqueeze(0), out["x_hat"])
        out_ssim = compute_msssim(img.unsqueeze(0), out["x_hat"])
        psnr_merged.append(out_psnr)
        ssim_merged.append(out_ssim)
        bpp_merged.append(out_bpp)
        compression_ratio["MERGED"] = compressed_size_in_bytes / img_size_in_bytes

    print("Running raw dataset")
    # Individual channels
    for img in tqdm(test_data_raw):
        # LICOS 1C
        img_size_in_bytes = 8 * img.shape[0] * img.shape[1] * img.shape[2]
        print("Original raw image size: ", img_size_in_bytes, " bytes")
        out, reconstructed, diff, compressed_size_in_bytes = process_img(img, net_raw)
        out_bpp = compute_bpp(out)
        out_psnr = compute_psnr(img.unsqueeze(0), out["x_hat"])
        out_ssim = compute_msssim(img.unsqueeze(0), out["x_hat"])
        psnr_raw.append(out_psnr)
        ssim_raw.append(out_ssim)
        bpp_raw.append(out_bpp)
        compression_ratio["RAW"] = compressed_size_in_bytes / img_size_in_bytes

        PIL_img = Image.fromarray(np.uint8(img.squeeze() * 255), "L")

        # JPEG, BPP Matched
        rec_jpeg, bpp_jpeg, psnr_val, mssim_val = find_closest_bpp(out_bpp, PIL_img)
        psnr_bbp_matched.append(psnr_val)
        ssim_bbp_matched.append(mssim_val)
        bpp_bbp_matched.append(bpp_jpeg)
        compression_ratio["JPEG_BPP"] = (
            bpp_jpeg * img.shape[1] * img.shape[2]
        ) / img_size_in_bytes

        # JPEG, PSNR Matched
        rec_jpeg, bpp_jpeg, psnr_val, mssim_val = find_closest_psnr(out_psnr, PIL_img)
        psnr_psnr_matched.append(psnr_val)
        ssim_psnr_matched.append(mssim_val)
        bpp_psnr_matched.append(bpp_jpeg)
        compression_ratio["JPEG_PSNR"] = (
            bpp_jpeg * img.shape[1] * img.shape[2]
        ) / img_size_in_bytes

        # JPEG, SSIM Matched
        rec_jpeg, bpp_jpeg, psnr_val, mssim_val = find_closest_msssim(out_ssim, PIL_img)
        psnr_ssim_matched.append(psnr_val)
        ssim_ssim_matched.append(mssim_val)
        bpp_ssim_matched.append(bpp_jpeg)
        compression_ratio["JPEG_MSSSIM"] = (
            bpp_jpeg * img.shape[1] * img.shape[2]
        ) / img_size_in_bytes

    print("Type \t PSNR \t MSSSIM \t BPP")
    print("RAW", np.mean(psnr_raw), np.mean(ssim_raw), np.mean(bpp_raw))
    print("MERGED", np.mean(psnr_merged), np.mean(ssim_merged), np.mean(bpp_merged))
    print(
        "JPEG BPP",
        np.mean(psnr_bbp_matched),
        np.mean(ssim_bbp_matched),
        np.mean(bpp_bbp_matched),
    )
    print(
        "JPEG PSNR",
        np.mean(psnr_psnr_matched),
        np.mean(ssim_psnr_matched),
        np.mean(bpp_psnr_matched),
    )
    print(
        "JPEG MSSIM",
        np.mean(psnr_ssim_matched),
        np.mean(ssim_ssim_matched),
        np.mean(bpp_ssim_matched),
    )

    results_df = results_df.append(
        {
            "Type": "RAW",
            "Seed": seed,
            "PSNR": np.mean(psnr_raw),
            "SSIM": np.mean(ssim_raw),
            "BPP": np.mean(bpp_raw),
            "Compression": compression_ratio["RAW"],
        },
        ignore_index=True,
    )
    results_df = results_df.append(
        {
            "Type": "MERGED",
            "Seed": seed,
            "PSNR": np.mean(psnr_merged),
            "SSIM": np.mean(ssim_merged),
            "BPP": np.mean(bpp_merged),
            "Compression": compression_ratio["MERGED"],
        },
        ignore_index=True,
    )
    results_df = results_df.append(
        {
            "Type": "JPEG BPP",
            "Seed": seed,
            "PSNR": np.mean(psnr_bbp_matched),
            "SSIM": np.mean(ssim_bbp_matched),
            "BPP": np.mean(bpp_bbp_matched),
            "Compression": compression_ratio["JPEG_BPP"],
        },
        ignore_index=True,
    )
    results_df = results_df.append(
        {
            "Type": "JPEG PSNR",
            "Seed": seed,
            "PSNR": np.mean(psnr_psnr_matched),
            "SSIM": np.mean(ssim_psnr_matched),
            "BPP": np.mean(bpp_psnr_matched),
            "Compression": compression_ratio["JPEG_PSNR"],
        },
        ignore_index=True,
    )
    results_df = results_df.append(
        {
            "Type": "JPEG SSIM",
            "Seed": seed,
            "PSNR": np.mean(psnr_ssim_matched),
            "SSIM": np.mean(ssim_ssim_matched),
            "BPP": np.mean(bpp_ssim_matched),
            "Compression": compression_ratio["JPEG_MSSSIM"],
        },
        ignore_index=True,
    )

# Compute mean over different seeds
results_df = results_df.groupby(["Type"]).mean().reset_index()
results_df.to_csv("results.csv", index=False)

# Print results
print(results_df)
