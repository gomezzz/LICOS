from torch.nn.functional import interpolate
import torch

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
        downsample_mode (str, optional): "nearest", "bilinear", "bicubic".
        If None, pixels are just dicarded. Defaults to blinear.
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
