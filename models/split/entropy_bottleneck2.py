from typing import List

import torch
from compressai.latent_codecs import EntropyBottleneckLatentCodec
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN1
from torch import nn

from models.mobilenetv3.mobilenetv3 import MobileNetV3
from models.split.channel_bottleneck import MobileNetV3Decoder
from models.split.split_model import SplitModel


class MV3EntropyBottleneck2(SplitModel):
    def __init__(self, base_model: MobileNetV3, bottleneck_ratio: float,
                 split_position: int, bottleneck_position: int, compression_parameter: float, **kwargs):
        super().__init__()
        self.base_model = base_model
        self.split_position = split_position
        self.bottleneck_position = bottleneck_position
        self.num_classes = base_model.classifier[3].out_features
        self.compression_parameter = compression_parameter
        bottleneck_channels = base_model.cfgs[self.bottleneck_position - 1][2]
        original_channels = base_model.cfgs[self.split_position - 1][2]
        encoder_layers_pre = list(base_model.features[:self.split_position])
        decoder_layers = nn.Sequential(*list(base_model.features[self.split_position:]))

        print("Building MV3ChannelBottleneck with split position: ", self.split_position)
        # print("Original channels: ", original_channels)
        print("Bottleneck channels: ", bottleneck_channels)

        self.encoder = MobileNetV3VanillaEncoder(layers_pre=encoder_layers_pre,
                                                 original_channels=original_channels,
                                                 compression_parameter=compression_parameter)
        self.decoder = MobileNetV3Decoder(layers=decoder_layers,
                                          conv=base_model.conv,
                                          avgpool=base_model.avgpool,
                                          classifier=base_model.classifier,
                                          original_channels=original_channels,
                                          bottleneck_ratio=-1)

    def forward(self, x):
        output = {}
        pixels = x.shape[-1] * x.shape[-2] * x.shape[-3]
        output = self.encoder(x)
        output['bpp'] = output['likelihoods']['y'].log2().sum() / pixels
        output['compression_loss'] = -self.compression_parameter * output['bpp']
        output['y_hat'] = self.decoder(output['y_hat'])

        return output

    def compress(self, x):
        pixels = x.shape[-1] * x.shape[-2] * x.shape[-3]
        output = self.encoder(x, compress=True)

        output['num_bytes'] = sum([len(s) for s in output['strings'][0]]) / len(output['strings'][0])
        output['bpp'] = output['num_bytes'] / pixels
        return output


class MobileNetV3VanillaEncoder(nn.Module):
    def __init__(self, layers_pre: List, original_channels: int, compression_parameter: float):
        super().__init__()
        self.original_channels = original_channels
        N = 128

        # self.bottleneck_channels = int(self.bottleneck_ratio * self.bottleneck_channels)
        entropy_bottleneck = EntropyBottleneck(N, filters=(8, 8, 8, 8))
        self.codec = EntropyBottleneckLatentCodec(entropy_bottleneck=entropy_bottleneck)
        self.compression_parameter = compression_parameter
        self.layers_pre = nn.Sequential(
            nn.Conv2d(3, N, 5, 2, 2),
            GDN1(N),
            nn.Conv2d(N, N, 5, 2, 2),
            GDN1(N),
            nn.Conv2d(N, N, 5, 2, 2),
            GDN1(N),
            nn.Conv2d(N, N, 5, 2, 2),
        )

        self.layers_post = nn.Sequential(
            nn.ConvTranspose2d(N, N, 5, 2, 2, 1),
            GDN1(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, 2, 2, 1),
            GDN1(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, 2, 2, 1),
            GDN1(N, inverse=True),
            nn.ConvTranspose2d(N, 3, 5, 2, 2, 1),
            *layers_pre
        )

    def forward(self, x, compress=False):
        pixels = x.shape[-1] * x.shape[-2] * x.shape[-3]
        x = self.layers_pre(x)

        if compress:
            x = self.codec.compress(x)
        else:
            x = self.codec(x)
            x['bpp'] = x['likelihoods']['y'].log2().sum() / pixels
            x['compression_loss'] = -self.compression_parameter * x['bpp']

        x['y_hat'] = self.layers_post(x['y_hat'])

        return x
