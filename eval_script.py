print("Loading imports...")
import sys

sys.path.append("./licos/")

import torch

from tqdm import tqdm
import numpy as np
from dotmap import DotMap
import toml
from PIL import Image
import io

import math
from pytorch_msssim import ms_ssim

from licos.model_utils import get_model
from licos.l0_image_folder import L0ImageFolder
from licos.l0_utils import DN_MAX
from licos.utils import get_savepath_str

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset = "/home/pablo/raw_test/"

cfg = DotMap(toml.load("cfg/l0.toml"), _dynamic=False)
seed = 42
checkpoint_path = (
    "results/bmshj2018-factorizedqual=1_l0=raw_seed=" + str(seed) + ".pth.tar"
)
checkpoint_merged_path = (
    "results/bmshj2018-factorizedqual=1_l0=merged_seed=" + str(seed) + ".pth.tar"
)


def pillow_encode(img, fmt="jpeg", quality=10):
    tmp = io.BytesIO()
    img.save(tmp, format=fmt, quality=quality)
    tmp.seek(0)
    filesize = tmp.getbuffer().nbytes
    bpp = filesize * float(8) / (img.size[0] * img.size[1])
    rec = Image.open(tmp)
    return rec, bpp


def psnr(a, b):
    a = np.asarray(a).astype(np.float32)
    b = np.asarray(b).astype(np.float32)
    mse = np.mean(np.square(a - b))
    return 20 * math.log10(255.0) - 10.0 * math.log10(mse)


def mssim(a, b):
    a = torch.from_numpy(np.asarray(a).astype(np.float32))
    b = torch.from_numpy(np.asarray(b).astype(np.float32))
    if len(a.shape) == 2:
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
    a = a.permute(2, 0, 1).unsqueeze(0)
    b = b.permute(2, 0, 1).unsqueeze(0)
    return ms_ssim(a, b, data_range=255.0).item()


def find_closest_bpp(target, img, fmt="jpeg"):
    lower = 0
    upper = 100
    prev_mid = upper
    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        rec, bpp = pillow_encode(img, fmt=fmt, quality=int(mid))
        if bpp > target:
            upper = mid - 1
        else:
            lower = mid
    psnr_val = psnr(rec, img)
    msssim_val = mssim(rec, img)
    return rec, bpp, psnr_val, msssim_val


def find_closest_psnr(target, img, fmt="jpeg"):
    lower = 0
    upper = 100
    prev_mid = upper

    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        prev_mid = mid
        rec, bpp = pillow_encode(img, fmt=fmt, quality=int(mid))
        psnr_val = psnr(rec, img)
        msssim_val = mssim(rec, img)
        if psnr_val > target:
            upper = mid - 1
        else:
            lower = mid
    return rec, bpp, psnr_val, msssim_val


def find_closest_msssim(target, img, fmt="jpeg"):
    lower = 0
    upper = 100
    prev_mid = upper

    for i in range(10):
        mid = (upper - lower) / 2 + lower
        if int(mid) == int(prev_mid):
            break
        prev_mid = mid
        rec, bpp = pillow_encode(img, fmt=fmt, quality=int(mid))
        msssim_val = mssim(rec, img)
        psnr_val = psnr(rec, img)
        if msssim_val > target:
            upper = mid - 1
        else:
            lower = mid
    return rec, bpp, psnr_val, msssim_val


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.0).item()


def compute_bpp(out_net):
    size = out_net["x_hat"].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(
        torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        for likelihoods in out_net["likelihoods"].values()
    ).item()


def process_img(img, net):
    with torch.no_grad():
        out_net = net.forward(img.unsqueeze(0))
    out_net["x_hat"].clamp_(0, 1)
    # print(out_net.keys())
    out_net["x_hat"] = out_net["x_hat"][..., : img.shape[1], : img.shape[2]]
    reconstructed = out_net["x_hat"].squeeze().cpu()
    diff = torch.mean((out_net["x_hat"] - img).abs(), axis=1).squeeze().cpu()
    return out_net, reconstructed, diff


print("Loading models...")
net_raw = get_model(cfg.model, False, 1, cfg.model_quality)
print(f"Parameters: {sum(p.numel() for p in net_raw.parameters())}")
checkpoint = torch.load(checkpoint_path, map_location=device)
net_raw.load_state_dict(checkpoint["state_dict"])
net_raw.update()

net_merged = get_model(cfg.model, False, 13, cfg.model_quality)
print(f"Parameters: {sum(p.numel() for p in net_merged.parameters())}")
checkpoint = torch.load(checkpoint_merged_path, map_location=device)
net_merged.load_state_dict(checkpoint["state_dict"])
net_merged.update()

print("Loading data...")
cfg.l0_train_test_split = 0.5  # for testing
test_data_raw = L0ImageFolder(
    dataset,
    cfg.seed,
    cfg.l0_train_test_split,
    "raw",
    cfg.l0_target_resolution_merged_m,
    split="test",
)
test_data_merged = L0ImageFolder(
    dataset,
    cfg.seed,
    cfg.l0_train_test_split,
    "merged",
    cfg.l0_target_resolution_merged_m,
    split="test",
)

print("Computing metrics")
psnr_raw, ssim_raw, bpp_raw = [], [], []
psnr_merged, ssim_merged, bpp_merged = [], [], []

psnr_bbp_matched, ssim_bbp_matched, bpp_bbp_matched = [], [], []
psnr_psnr_matched, ssim_psnr_matched, bpp_psnr_matched = [], [], []
psnr_ssim_matched, ssim_ssim_matched, bpp_ssim_matched = [], [], []

print("Running merged dataset")
# Merged channels
for img in tqdm(test_data_merged):
    # LICOS 13C
    out, reconstructed, diff = process_img(img, net_merged)
    out_bpp = compute_bpp(out)
    out_psnr = compute_psnr(img.unsqueeze(0), out["x_hat"])
    out_ssim = compute_msssim(img.unsqueeze(0), out["x_hat"])
    psnr_merged.append(out_psnr)
    ssim_merged.append(out_ssim)
    bpp_merged.append(out_bpp)

print("Running raw dataset")
# Individual channels
for img in tqdm(test_data_raw):
    # LICOS 1C
    out, reconstructed, diff = process_img(img, net_raw)
    out_bpp = compute_bpp(out)
    out_psnr = compute_psnr(img.unsqueeze(0), out["x_hat"])
    out_ssim = compute_msssim(img.unsqueeze(0), out["x_hat"])
    psnr_raw.append(out_psnr)
    ssim_raw.append(out_ssim)
    bpp_raw.append(out_bpp)

    PIL_img = Image.fromarray(np.uint8(img.squeeze() * 255), "L")

    # JPEG, BPP Matched
    rec_jpeg, bpp_jpeg, psnr_val, mssim_val = find_closest_bpp(out_bpp, PIL_img)
    psnr_bbp_matched.append(psnr_val)
    ssim_bbp_matched.append(mssim_val)
    bpp_bbp_matched.append(bpp_jpeg)

    # JPEG, PSNR Matched
    rec_jpeg, bpp_jpeg, psnr_val, mssim_val = find_closest_psnr(out_psnr, PIL_img)
    psnr_psnr_matched.append(psnr_val)
    ssim_psnr_matched.append(mssim_val)
    bpp_psnr_matched.append(bpp_jpeg)

    # JPEG, SSIM Matched
    rec_jpeg, bpp_jpeg, psnr_val, mssim_val = find_closest_msssim(out_ssim, PIL_img)
    psnr_ssim_matched.append(psnr_val)
    ssim_ssim_matched.append(mssim_val)
    bpp_ssim_matched.append(bpp_jpeg)


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
