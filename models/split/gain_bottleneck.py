from typing import List

import torch
from compressai.latent_codecs import GainHyperLatentCodec
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN1
from torch import nn

from models.mobilenetv3.mobilenetv3 import MobileNetV3
from models.split.channel_bottleneck import MobileNetV3Decoder
from models.split.split_model import SplitModel


class MV3GainBottleneck(SplitModel):
    def __init__(self, base_model: MobileNetV3, bottleneck_ratio: float,
                 split_position: int, bottleneck_position: int, beta: float, **kwargs):
        super().__init__()
        self.base_model = base_model
        self.split_position = split_position
        self.bottleneck_position = bottleneck_position
        self.num_classes = base_model.classifier[3].out_features
        self.beta = beta
        bottleneck_channels = base_model.cfgs[self.bottleneck_position - 1][2]
        original_channels = base_model.cfgs[self.split_position - 1][2]
        encoder_layers_pre = list(base_model.features[:self.bottleneck_position])
        encoder_layers_post = list(base_model.features[self.bottleneck_position:self.split_position])
        decoder_layers = nn.Sequential(*list(base_model.features[self.split_position:]))

        print("Building MV3GainBottleneck with split position: ", self.split_position)
        # print("Original channels: ", original_channels)
        print("Bottleneck channels: ", bottleneck_channels)

        self.encoder = MobileNetV3VanillaEncoder(layers_pre=encoder_layers_pre,
                                                 layers_post=encoder_layers_post,
                                                 bottleneck_channels=bottleneck_channels,
                                                 bottleneck_ratio=bottleneck_ratio,
                                                 beta=beta)
        self.decoder = MobileNetV3Decoder(layers=decoder_layers,
                                          conv=base_model.conv,
                                          avgpool=base_model.avgpool,
                                          classifier=base_model.classifier,
                                          original_channels=original_channels,
                                          bottleneck_ratio=-1)

    def forward(self, x):
        output = {}
        pixels = x.shape[0] * x.shape[-1] * x.shape[-2] * x.shape[-3]
        output = self.encoder(x)
        output['compression_loss'] = -self.beta * output['bpp']
        output['y_hat'] = self.decoder(output['y_hat'])

        return output

    def compress(self, x):
        pixels = x.shape[-1] * x.shape[-2] * x.shape[-3]
        output = self.encoder(x, compress=True)
        output['num_bytes'] = sum([len(s) for s in output['strings'][0]]) / len(output['strings'][0])
        output['bpp'] = output['num_bytes'] / pixels
        return output


class MobileNetV3VanillaEncoder(nn.Module):
    def __init__(self, layers_pre: List, layers_post: List,
                 bottleneck_channels: int, bottleneck_ratio: float,
                 beta: float):
        super().__init__()
        self.layers_pre = nn.Sequential(*layers_pre)
        self.layers_post = nn.Sequential(*layers_post)
        self.bottleneck_ratio = bottleneck_ratio
        self.bottleneck_channels = bottleneck_channels
        self.tier = 0

        entropy_bottleneck = EntropyBottleneck(self.bottleneck_channels, filters=(8, 8, 8, 8))
        self.codec = GainHyperLatentCodec(entropy_bottleneck=entropy_bottleneck)
        self.codec.entropy_bottleneck.update()
        self.beta = beta
        self.num_betas = 10
        self.gain = torch.nn.init.uniform_(torch.randn( self.num_betas, self.bottleneck_channels, 1, 1)).to('cuda')
        self.inv_gain = torch.nn.init.uniform_(torch.randn( self.num_betas, self.bottleneck_channels, 1, 1)).to('cuda')

        # self.gain.requires_grad = False
        # self.inv_gain.requires_grad = False

    def forward(self, x, compress=False, tier=5):
        pixels = x.shape[0] * x.shape[-1] * x.shape[-2] * x.shape[-3]
        x = self.layers_pre(x)

        betas = torch.linspace(0, 1.0,  self.num_betas).to('cuda')
        if self.training:
            self.tier = torch.randint(0, self.num_betas, (1,)).to('cuda')

        else:
            self.tier = tier

        if compress:
            x = self.codec.compress(x, self.gain[self.tier], self.inv_gain[self.tier])
        else:
            x = self.codec(x, self.gain[self.tier], self.inv_gain[self.tier])
            x['bpp'] = x['likelihoods']['z'].log2().sum() / pixels
            x['compression_loss'] = -betas[self.tier] * x['bpp']
            # print(x['bpp'], self.tier, x['compression_loss'])
            print(x['compression_loss'])

        x['y_hat'] = self.layers_post(x['params'])

        return x
