import torch
from compressai.latent_codecs import EntropyBottleneckLatentCodec
from compressai.layers import GDN1
from torch import nn

from models.mobilenetv3.mobilenetv3 import MobileNetV3
from models.split.channel_bottleneck import MobileNetV3Decoder


class MV3EntropyBottleneck(nn.Module):
    def __init__(self, base_model: MobileNetV3, bottleneck_ratio: float, split_position: int, **kwargs):
        super().__init__()
        self.base_model = base_model
        self.split_position = split_position
        self.num_classes = base_model.classifier[3].out_features
        original_channels = base_model.cfgs[self.split_position][2]
        encoder_layers = nn.Sequential(*list(base_model.features[:self.split_position]))
        decoder_layers = nn.Sequential(*list(base_model.features[self.split_position:]))

        print("Building MV3ChannelBottleneck with split position: ", self.split_position)
        print("Original channels: ", original_channels)
        print("Bottleneck channels: ", int(bottleneck_ratio * original_channels))

        self.encoder = MobileNetV3VanillaEncoder(encoder_layers,
                                                 original_channels=original_channels,
                                                 bottleneck_ratio=bottleneck_ratio)
        self.decoder = MobileNetV3Decoder(layers=decoder_layers,
                                          conv=base_model.conv,
                                          avgpool=base_model.avgpool,
                                          classifier=base_model.classifier,
                                          original_channels=original_channels,
                                          bottleneck_ratio=bottleneck_ratio)


    def forward(self, x):
        output = {}
        output = self.encoder(x)
        if not self.training:
            output['num_bytes'] = sum([len(s) for s in output['strings'][0]])/len(output['strings'][0])
        output['y_hat'] = self.decoder(output['y_hat'])
        output['compression_loss'] = output['likelihoods'].log2().mean()
        return output



class MobileNetV3VanillaEncoder(nn.Module):
    def __init__(self, layers: nn.Sequential, original_channels: int, bottleneck_ratio: float):
        super().__init__()
        self.layers = layers
        self.bottleneck_ratio = bottleneck_ratio
        self.original_channels = original_channels
        self.bottleneck_channels = int(self.bottleneck_ratio * self.original_channels)
        self.codec = EntropyBottleneckLatentCodec(channels=self.bottleneck_channels)
        self.codec.entropy_bottleneck.update()

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