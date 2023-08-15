import torch
from compressai.latent_codecs import EntropyBottleneckLatentCodec
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN1
from torch import nn

from models.mobilenetv3.mobilenetv3 import MobileNetV3
from models.split.channel_bottleneck import MobileNetV3Decoder


class MV3Precompressor(nn.Module):
    def __init__(self, base_model: MobileNetV3, compression_parameter=0.0, **kwargs):
        super().__init__()
        self.base_model = base_model
        self.num_classes = base_model.classifier[3].out_features
        self.compression_parameter = compression_parameter

        self.encoder = Precompressor()
        self.decoder = self.base_model

    def forward(self, x):
        output = self.encoder(x)
        if self.training:
            output['compression_loss'] = -self.compression_parameter * output['likelihoods']['y'].log2().mean()
        else:
            output['num_bytes'] = sum([len(s) for s in output['strings'][0]]) / len(output['strings'][0])
        output['y_hat'] = self.decoder(output['y_hat'])

        return output


class Precompressor(nn.Module):
    def __init__(self, N=128):
        super().__init__()
        entropy_bottleneck = EntropyBottleneck(N, filters=(8, 8, 8, 8))
        self.codec = EntropyBottleneckLatentCodec(entropy_bottleneck)
        self.codec.entropy_bottleneck.update()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, N, 5, 2, 2),
            GDN1(N),
            nn.Conv2d(N, N, 5, 2, 2),
            GDN1(N),
            nn.Conv2d(N, N, 5, 2, 2),
            GDN1(N),
            nn.Conv2d(N, N, 5, 2, 2),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(N, N, 5, 2, 2, 1),
            GDN1(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, 2, 2, 1),
            GDN1(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, 2, 2, 1),
            GDN1(N, inverse=True),
            nn.ConvTranspose2d(N, 3, 5, 2, 2, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        if self.training:
            x = self.codec(x)
        else:
            x = self.codec.compress(x)

        x['y_hat'] = self.decoder(x['y_hat'])
        return x
