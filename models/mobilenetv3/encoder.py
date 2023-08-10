import torch
from compressai.latent_codecs import EntropyBottleneckLatentCodec
from compressai.layers import GDN1
from torch import nn


def get_encoder(encoder: str):
    encoder_dict = {
        "vanilla": MobileNetV3VanillaEncoder,
        "gdn": MobileNetV3GdnEncoder,
    }
    return encoder_dict[encoder]




class MobileNetV3GdnEncoder(nn.Module):
    def __init__(self, layers: nn.Sequential, original_channels: int, bottleneck_ratio: float):
        super().__init__()
        self.layers = layers
        self.bottleneck_ratio = bottleneck_ratio
        self.original_channels = original_channels
        self.bottleneck_channels = int(self.bottleneck_ratio * self.original_channels)
        N = self.bottleneck_channels
        N2 = self.original_channels
        self.codec = EntropyBottleneckLatentCodec(channels=N)
        self.encoder = nn.Sequential(
            nn.Conv2d(N2, N, 3, 1, 0),
            GDN1(N),
            nn.Conv2d(N, N, 3, 1, 0),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(N, N, 3, 1, 1, 0),
            GDN1(N, inverse=True),
            nn.ConvTranspose2d(N, N2, 3, 1, 1, 0)
        )


    def forward(self, x):
        x = self.layers(x)
        if self.bottleneck_channels > 0 and self.bottleneck_ratio <= 1:
            x[:, self.bottleneck_channels:, ::] = 0
            x = x[:, :self.bottleneck_channels, ::]
        x = self.encoder(x)
        if self.training:
            x = self.codec(x)
        else:
            x= self.codec.compress(x)
        x['y_hat'] = self.decoder(x['y_hat'])
        return x

