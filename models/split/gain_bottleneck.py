import math
from typing import List

import numpy as np
import torch
from compressai.latent_codecs import GainHyperLatentCodec, GainHyperpriorLatentCodec
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN1
from torch import nn

from models.compression import GainHyperpriorLatentCodecFixed
from models.mobilenetv3.mobilenetv3 import MobileNetV3
from models.split.channel_bottleneck import MobileNetV3Decoder
from models.split.split_model import SplitModel


class MV3GainBottleneck(SplitModel):
    def __init__(self, base_model: MobileNetV3, bottleneck_ratio: float,
                 split_position: int, bottleneck_position: int, num_betas=6, max_beta=1.5, **kwargs):
        super().__init__()
        self.base_model = base_model
        self.split_position = split_position
        self.bottleneck_position = bottleneck_position
        self.num_classes = base_model.classifier[3].out_features
        bottleneck_channels = base_model.cfgs[self.bottleneck_position - 1][2]
        original_channels = base_model.cfgs[self.split_position - 1][2]
        encoder_layers_pre = list(base_model.features[:self.bottleneck_position])
        encoder_layers_post = list(base_model.features[self.bottleneck_position:self.split_position])
        decoder_layers = nn.Sequential(*list(base_model.features[self.split_position:]))

        print("Building MV3GainBottleneck with split position: ", self.split_position)
        # print("Original channels: ", original_channels)
        print("Bottleneck channels: ", bottleneck_channels)

        self.encoder = MobileNetV3GainEncoder(layers_pre=encoder_layers_pre,
                                              layers_post=encoder_layers_post,
                                              bottleneck_channels=bottleneck_channels,
                                              bottleneck_ratio=bottleneck_ratio,
                                              num_betas=num_betas, max_beta=max_beta)
        self.decoder = MobileNetV3Decoder(layers=decoder_layers,
                                          conv=base_model.conv,
                                          avgpool=base_model.avgpool,
                                          classifier=base_model.classifier,
                                          original_channels=original_channels,
                                          bottleneck_ratio=-1)

    def forward(self, x, tier=0):
        output = {}
        pixels = x.shape[0] * x.shape[-1] * x.shape[-2] * x.shape[-3]
        output = self.encoder(x, tier=tier)
        output['y_hat'] = self.decoder(output['y_hat'])

        return output

    def compress(self, x, tier=0):
        pixels = x.shape[-1] * x.shape[-2] * x.shape[-3]
        output = self.encoder(x, compress=True, tier=tier)
        output['num_bytes'] = sum([len(s) for s in output['strings'][0]]) / len(output['strings'][0])
        output['num_bytes'] += sum([len(s) for s in output['strings'][1]]) / len(output['strings'][1])
        output['bpp'] = output['num_bytes'] / pixels
        return output


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):  # pylint: disable=W0622
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class MobileNetV3GainEncoder(nn.Module):
    def __init__(self, layers_pre: List, layers_post: List,
                 bottleneck_channels: int, bottleneck_ratio: float, num_betas=6, max_beta=1.5, **kwargs):
        super().__init__()
        self.layers_pre = nn.Sequential(*layers_pre)
        self.layers_post = nn.Sequential(*layers_post)
        self.bottleneck_ratio = bottleneck_ratio
        self.bottleneck_channels = bottleneck_channels
        self.tier = 0

        self.codec = GainHyperpriorLatentCodecFixed()
        self.num_betas = num_betas
        self.betas = torch.tensor([0.001*(5**i) for i in range(num_betas)]).to('cuda')
        self.num_betas = len(self.betas)

        self.gain = torch.nn.Parameter(torch.ones(self.num_betas, self.bottleneck_channels, 1, 1).to('cuda'),
                                       requires_grad=True)
        self.z_gain = torch.nn.Parameter(torch.ones(self.num_betas, self.bottleneck_channels, 1, 1).to('cuda'),
                                       requires_grad=True)

        self.inv_gain = torch.nn.Parameter(torch.ones(self.num_betas, self.bottleneck_channels, 1, 1).to('cuda'),
                                       requires_grad=True)
        self.inv_z_gain = torch.nn.Parameter(torch.ones(self.num_betas, self.bottleneck_channels, 1, 1).to('cuda'),
                                       requires_grad=True)

        # self.gain_scale = torch.nn.Parameter(torch.randn(self.num_betas, self.bottleneck_channels, 1, 1).to('cuda'))
        #
        # self.betas = torch.linspace(0.05, max_beta, self.num_betas).to('cuda')
        # self.betas = [0.001, 0.005, 0.01, 0.05, 0.1]
        # self.gain.requires_grad = False
        # self.inv_gain.requires_grad = False
        self.update()

    def update(self):
        self.codec.latent_codec['y'].gaussian_conditional.update_scale_table(get_scale_table())
        self.codec.latent_codec['y'].gaussian_conditional.update()
        self.codec.latent_codec['hyper'].entropy_bottleneck.update()

    def forward(self, x, compress=False, tier=0):
        pixels = x.shape[0] * x.shape[-1] * x.shape[-2] * x.shape[-3]
        x = self.layers_pre(x)

        if self.training:
            self.tier = torch.randint(0, self.num_betas, (1,)).to('cuda')
            # p = np.array(list(range(self.num_betas, 0, -1)))
            # p = p/p.sum()
            # self.tier = np.random.choice(list(range(self.num_betas)), size=1, p=p)


        else:
            self.tier = tier

        inv_gain = 1 / self.gain[self.tier]
        inv_z_gain = 1 / self.gain[self.tier]
        # inv_gain = inv_gain * self.gain_scale[self.tier]
        # self.gain_scale[self.tier]
        # print(inv_gain.shape, self.gain_scale.shape, self.gain[self.tier].shape)
        # inv_gain = self.inv_gain[self.tier]

        if compress:
            self.update()
            x = self.codec.compress(x, self.gain[self.tier], self.z_gain[self.tier],
                                    self.inv_gain[self.tier], self.inv_z_gain[self.tier])
        else:
            # print(self.z_gain[self.tier].shape, self.gain[self.tier].shape, inv_gain.shape, inv_z_gain.shape)
            # exit()

            x = self.codec(x, self.gain[self.tier], self.z_gain[self.tier],
                                    self.inv_gain[self.tier], self.inv_z_gain[self.tier])
            x['bpp'] = (x['likelihoods']['y'].log2().sum() + x['likelihoods']['z'].log2().sum()) / pixels
            x['compression_loss'] = -self.betas[self.tier].item() * x['bpp']
            # print(x['bpp'], self.tier, x['compression_loss'])
            # print(x['compression_loss'])

        x['y_hat'] = self.layers_post(x['y_hat'])

        return x
