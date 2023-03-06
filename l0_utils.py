from torch.nn import Upsample

# Sentinel-2 band names
BAND_LIST=["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B8A", "B10", "B11", "B12"]

# Spatial resolution of various bands
BAND_SPATIAL_RESOLUTION_DICT=dict(zip(BAND_LIST, [60, 10, 10, 10, 20, 20, 20, 10, 60, 20, 60, 20, 20]))


def image_band_upsample(img_band, band_name, upsample_mode="bilinear"): 
    """Upsample an image band to a target spatial resolution through an upsample mode.

    Args:
        img_band (torch.tensor): image band.
        band_name (string): band name.
        upsample_mode (string, optional): "nearest", "bilinear", "bicubic". Defaults to blinear.

    Raises:
        ValueError: unsupported band name.
        ValueError: unsupported upsample mode.

    Returns:
        torch.tensor: upsampled band.
    """
    upsample_factor=BAND_SPATIAL_RESOLUTION_DICT[band_name]/10
    if upsample_factor  <= 1.0:
        return img_band
    
    if not(band_name in BAND_LIST):
        raise ValueError("Unsupported band name: "+band_name+".")
        
    if BAND_SPATIAL_RESOLUTION_DICT[band_name] == 60:
        #Recreating a square image (TODO: this is a horrible appproach. To be improved.)
        img_band=img_band[:,::3]
    
    if not(upsample_mode in ['nearest', 'bilinear', 'bicubic']):
        raise ValueError("Upsample mode "+upsample_mode+" not supported. Please, choose among: ""nearest"", ""bilinear"", ""bicubic"".")

    if upsample_factor != int(upsample_factor):
        print("Warning. Upsample factor truncanted from "+upsample_factor+" to "+int(upsample_factor)+".")

    upsample_factor=int(upsample_factor)
    if upsample_mode == "nearest":
        upsample_method=Upsample(scale_factor=upsample_factor, mode=upsample_mode) #, align_corners=True)
    else:
        upsample_method=Upsample(scale_factor=upsample_factor, mode=upsample_mode, align_corners=True)
        
    with torch.no_grad():
        return upsample_method(img_band.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)



