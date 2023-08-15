import torch
from compressai.latent_codecs import EntropyBottleneckLatentCodec
from compressai.layers import GDN1
from torch import nn

from models.mobilenetv3.mobilenetv3 import MobileNetV3
from models.split.split_model import SplitModel


class MV3ChannelBottleneck(SplitModel):
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
        pixels = x.shape[-1] * x.shape[-2] * x.shape[-3]
        x = self.encoder(x)
        output['num_bytes'] = x.shape[1]*x.shape[2]*x.shape[3]
        output['y_hat'] = self.decoder(x)
        output['bpp'] = output['num_bytes'] / pixels
        output['compression_loss'] = 0.0
        return output

    def compress(self, x):
        pixels = x.shape[-1] * x.shape[-2] * x.shape[-3]
        x = self.encoder(x)
        output = {'num_bytes': x.shape[1]*x.shape[2]*x.shape[3]}
        output['bpp'] = output['num_bytes'] / pixels
        return output


class MobileNetV3VanillaEncoder(nn.Module):
    def __init__(self, layers: nn.Sequential, original_channels: int, bottleneck_ratio: float):
        super().__init__()
        self.layers = layers
        self.bottleneck_ratio = bottleneck_ratio
        self.original_channels = original_channels
        self.bottleneck_channels = int(self.bottleneck_ratio * self.original_channels)

    def forward(self, x):
        x = self.layers(x)
        if self.bottleneck_channels > 0:
            x[:, self.bottleneck_channels:, ::] = 0
            x = x[:, :self.bottleneck_channels, ::]

        return x


class MobileNetV3Decoder(nn.Module):
    def __init__(self, layers, conv, avgpool, classifier, original_channels: int, bottleneck_ratio: float):
        super().__init__()
        self.layers = layers
        self.conv = conv
        self.avgpool = avgpool
        self.classifier = classifier
        self.bottleneck_ratio = bottleneck_ratio
        self.original_channels = original_channels


    def forward(self, x):
        original_size = self.layers[0].conv[0].in_channels
        bottleneck_channels = int(self.bottleneck_ratio * self.original_channels)
        if bottleneck_channels > 0 and self.bottleneck_ratio <= 1:
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

