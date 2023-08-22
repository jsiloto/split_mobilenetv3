import torch
from compressai.latent_codecs import EntropyBottleneckLatentCodec
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN1
from torch import nn

from models.mobilenetv3.mobilenetv3 import MobileNetV3
from models.split.channel_bottleneck import MobileNetV3Decoder
from models.split.split_model import SplitModel


class MV3Precompressor(SplitModel):
    def __init__(self, base_model: MobileNetV3, beta=0.0, **kwargs):
        super().__init__()
        self.base_model = base_model
        self.num_classes = base_model.classifier[3].out_features
        self.beta = beta

        self.encoder = Precompressor(beta=beta, N=128)
        self.decoder = self.base_model

    def forward(self, x):
        output = self.encoder(x)
        output['y_hat'] = self.decoder(output['y_hat'])
        return output

    def compress(self, x):
        output = self.encoder.compress(x)
        output['num_bytes'] = sum([len(s) for s in output['strings'][0]]) / len(output['strings'][0])
        return output

class Precompressor(nn.Module):
    def __init__(self, beta: float, N: int = 128):
        super().__init__()
        entropy_bottleneck = EntropyBottleneck(N, filters=(8, 8, 8, 8))
        self.beta = beta
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
        x = self.codec(x)
        x['y_hat'] = self.decoder(x['y_hat'])
        x['compression_loss'] = -self.beta * x['likelihoods']['y'].log2().mean()
        return x

    def compress(self, x):
        x = self.encoder(x)
        x = self.codec.compress(x)
        return x
