import io
import math
import torch
from PIL import Image
from pytorch_msssim import ms_ssim
import numpy as np


def pillow_encode(img, fmt="jpeg", quality=10):
    """Encode image with Pillow and return reconstructed image and bpp"""

    tmp = io.BytesIO()
    img.save(tmp, format=fmt, quality=quality)
    tmp.seek(0)
    filesize = tmp.getbuffer().nbytes
    bpp = filesize * float(8) / (img.size[0] * img.size[1])
    rec = Image.open(tmp)
    return rec, bpp


def psnr(a, b):
    """Compute PSNR between two images

    Args:
        a (np.array): First image
        b (np.array): Second image

    Returns:
        float: PSNR value
    """
    a = np.asarray(a).astype(np.float32)
    b = np.asarray(b).astype(np.float32)
    mse = np.mean(np.square(a - b))
    return 20 * math.log10(255.0) - 10.0 * math.log10(mse)


def mssim(a, b):
    """Compute MS-SSIM between two images

    Args:
        a (np.array): First image
        b (np.array): Second image

    Returns:
        float: MS-SSIM value
    """
    a = torch.from_numpy(np.asarray(a).astype(np.float32))
    b = torch.from_numpy(np.asarray(b).astype(np.float32))
    if len(a.shape) == 2:
        a = a.unsqueeze(2)
        b = b.unsqueeze(2)
    a = a.permute(2, 0, 1).unsqueeze(0)
    b = b.permute(2, 0, 1).unsqueeze(0)
    return ms_ssim(a, b, data_range=255.0).item()


def find_closest_bpp(target, img, fmt="jpeg"):
    """Find the closest bpp to the target bpp

    Args:
        target (float): Target bpp
        img (np.array): Image to encode
        fmt (str, optional): Encoding format. Defaults to "jpeg".

    Returns:
        tuple: Reconstructed image, bpp, PSNR, MS-SSIM
    """
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
    """Find the closest PSNR to the target PSNR

    Args:
        target (float): Target PSNR
        img (np.array): Image to encode
        fmt (str, optional): Encoding format. Defaults to "jpeg".

    Returns:
        tuple: Reconstructed image, bpp, PSNR, MS-SSIM
    """
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
    """Find the closest MS-SSIM to the target MS-SSIM

    Args:
        target (float): Target MS-SSIM
        img (np.array): Image to encode
        fmt (str, optional): Encoding format. Defaults to "jpeg".

    Returns:
        tuple: Reconstructed image, bpp, PSNR, MS-SSIM
    """
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
    """Compute PSNR between two images

    Args:
        a (torch.Tensor): First image
        b (torch.Tensor): Second image

    Returns:
        float: PSNR value
    """
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    """Compute MS-SSIM between two images

    Args:
        a (torch.Tensor): First image
        b (torch.Tensor): Second image

    Returns:
        float: MS-SSIM value
    """
    return ms_ssim(a, b, data_range=1.0).item()


def compute_bpp(out_net):
    """Compute bpp from the output of the network

    Args:
        out_net (dict): Output of the network

    Returns:
        float: bpp value
    """
    size = out_net["x_hat"].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(
        torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        for likelihoods in out_net["likelihoods"].values()
    ).item()


def process_img(img, net):
    """Process an image through the network

    Args:
        img (torch.Tensor): Image to process
        net (torch.nn.Module): Network to use

    Returns:
        tuple: Output of the network, reconstructed image, difference image, compressed size in bytes
    """
    with torch.no_grad():
        out_net = net.forward(img.unsqueeze(0))
        compressed_img = net.compress(img.unsqueeze(0))
    compressed_size_in_bytes = np.frombuffer(
        np.array(compressed_img["strings"]), dtype=np.uint8
    ).size
    out_net["x_hat"].clamp_(0, 1)
    # print(out_net.keys())
    out_net["x_hat"] = out_net["x_hat"][..., : img.shape[1], : img.shape[2]]
    reconstructed = out_net["x_hat"].squeeze().cpu()
    diff = torch.mean((out_net["x_hat"] - img).abs(), axis=1).squeeze().cpu()
    return out_net, reconstructed, diff, compressed_size_in_bytes
