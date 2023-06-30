from compressai.zoo import image_models
from compressai.entropy_models import EntropyBottleneck
from torch.nn import Conv2d, ConvTranspose2d


def get_model(model, pretrained, in_channels=3, quality=1):
    """Get models and change ther input channels.
       Supported models include: bmshj2018-factorized, bmshj2018-factorized-relu, bmshj2018-hyperprior,

    Args:
        model (str): model name.
        pretrained (bool): if True, pretrained version is used.
        in_channels (int, optional): number of input channels. Defaults to 3.
        quality (int, optional): CompressAI quality parameter.

    Returns:
        net: adapted model
    """
    net = image_models[model](quality=quality, pretrained=pretrained)
    if model in [
        "bmshj2018-factorized",
        "bmshj2018-factorized-relu",
        "bmshj2018-hyperprior",
    ]:
        ebn = EntropyBottleneck(
            channels=net.entropy_bottleneck.channels,
            filters=(in_channels, in_channels, 3, 3),
        )
        net.entropy_bottleneck = ebn

        net.g_a[0] = Conv2d(
            in_channels=in_channels,
            out_channels=net.g_a[0].out_channels,
            kernel_size=(net.g_a[0].weight.shape[2], net.g_a[0].weight.shape[3]),
            stride=net.g_a[0].stride,
            padding=net.g_a[0].padding,
        )
        net.g_s[6] = ConvTranspose2d(
            in_channels=net.g_s[6].in_channels,
            out_channels=in_channels,
            kernel_size=(net.g_s[6].weight.shape[2], net.g_s[6].weight.shape[3]),
            stride=net.g_s[6].stride,
            padding=net.g_s[6].padding,
            output_padding=net.g_s[6].output_padding,
        )

    else:
        raise ValueError("model: " + model + " not supported for raw data.")
    return net
