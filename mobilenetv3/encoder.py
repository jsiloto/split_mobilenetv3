import torch
from compressai.latent_codecs import EntropyBottleneckLatentCodec
from torch import nn

from mobilenetv3.layers import h_swish, h_sigmoid, InvertedResidual


def get_encoder(encoder: str):
    encoder_dict = {
        "vanilla": MobileNetV3VanillaEncoder,
        "gdn": MobileNetV3VanillaEncoder,
    }
    return encoder_dict[encoder]


class MobileNetV3VanillaEncoder(nn.Module):
    def __init__(self, layers: nn.Sequential, original_channels: int, bottleneck_ratio: float):
        super().__init__()
        self.layers = layers
        self.bottleneck_ratio = bottleneck_ratio
        self.original_channels = original_channels
        self.bottleneck_channels = int(self.bottleneck_ratio * self.original_channels)
        self.codec = EntropyBottleneckLatentCodec(channels=self.bottleneck_channels)

    def forward(self, x):
        x = self.layers(x)
        if self.bottleneck_channels > 0:
            x[:, self.bottleneck_channels:, ::] = 0
            x = x[:, :self.bottleneck_channels, ::]

        if self.training:
            x = self.codec(x)
        else:
            x = self.codec.compress(x)
        return x


class MobileNetV3Decoder(nn.Module):
    def __init__(self, layers, conv, avgpool, classifier, codec, original_channels: int, bottleneck_ratio: float):
        super().__init__()
        self.layers = layers
        self.conv = conv
        self.avgpool = avgpool
        self.classifier = classifier
        self.bottleneck_ratio = bottleneck_ratio
        self.original_channels = original_channels
        self.codec = codec
        print(layers)


    def forward(self, x):
        original_size = self.layers[0].conv[0].in_channels
        bottleneck_channels = int(self.bottleneck_ratio * self.original_channels)
        if self.bottleneck_ratio > 0:
            device = x.get_device()
            if device < 0:
                device = torch.device("cpu")
            zeros = torch.zeros(x.shape[0], original_size - bottleneck_channels, x.shape[2], x.shape[3]).to(device)
            x = torch.cat((x, zeros), dim=1)

        x = self.layers(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
