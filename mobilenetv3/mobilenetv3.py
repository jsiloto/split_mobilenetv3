"""
Creates a MobileNetV3 Model as defined in:
Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam. (2019).
Searching for MobileNetV3
arXiv preprint arXiv:1905.02244.
"""
import torch
import torch.nn as nn
import math
from compressai.latent_codecs import GainHyperpriorLatentCodec, HyperpriorLatentCodec, EntropyBottleneckLatentCodec

__all__ = ['mobilenetv3_large', 'mobilenetv3_small']

from mobilenetv3.encoder import MobileNetV3VanillaEncoder, MobileNetV3Decoder, get_encoder
from mobilenetv3.layers import _make_divisible, conv_3x3_bn, InvertedResidual, conv_1x1_bn, h_swish


class MobileNetV3(nn.Module):
    def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.,
                 split_position=1, bottleneck_ratio=-1, encoder="vanilla"):
        super(MobileNetV3, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.split_position = split_position
        self.bottleneck = nn.Identity()
        assert mode in ['large', 'small']
        assert split_position >= 1

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for layer_num, (k, t, c, use_se, use_hs, s) in enumerate(self.cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'large': 1280, 'small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[
            mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            nn.Linear(output_channel, num_classes),
        )

        original_channels = self.cfgs[self.split_position][2]
        encoder_layers = nn.Sequential(*list(self.features[:self.split_position]))
        decoder_layers = nn.Sequential(*list(self.features[self.split_position:]))

        self.encoder = get_encoder(encoder)(encoder_layers,
                                            original_channels=original_channels,
                                            bottleneck_ratio=bottleneck_ratio)
        self.decoder = MobileNetV3Decoder(layers=decoder_layers,
                                          conv=self.conv,
                                          avgpool=self.avgpool,
                                          classifier=self.classifier,
                                          codec=self.encoder.codec,
                                          original_channels=original_channels,
                                          bottleneck_ratio=bottleneck_ratio)

        self._initialize_weights()

    def forward(self, x):
        output = self.encoder(x)
        y_hat = self.decoder(output['y_hat'])
        output['y_hat'] = y_hat
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        # Dong et al 2022 initialization
        def dong_init(block):
            for m in block.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * math.sqrt(m.out_channels * m.in_channels)
                    m.weight.data.normal_(0, math.sqrt(2. / n))

        if self.split_position > 0:
            dong_init(self.encoder.layers[-1])
            dong_init(self.decoder.layers[0])


def mobilenetv3_large(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 0, 0, 1],
        [3, 4, 24, 0, 0, 2],
        [3, 3, 24, 0, 0, 1],
        [5, 3, 40, 1, 0, 2],
        [5, 3, 40, 1, 0, 1],
        [5, 3, 40, 1, 0, 1],
        [3, 6, 80, 0, 1, 2],
        [3, 2.5, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 2.3, 80, 0, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [3, 6, 112, 1, 1, 1],
        [5, 6, 160, 1, 1, 2],
        [5, 6, 160, 1, 1, 1],
        [5, 6, 160, 1, 1, 1]
    ]
    return MobileNetV3(cfgs, mode='large', **kwargs)


def mobilenetv3_small(**kwargs):
    """
    Constructs a MobileNetV3-Small model
    """
    cfgs = [
        # k, t, c, SE, HS, s
        [3, 1, 16, 1, 0, 2],
        [3, 4.5, 24, 0, 0, 2],
        [3, 3.67, 24, 0, 0, 1],
        [5, 4, 40, 1, 1, 2],
        [5, 6, 40, 1, 1, 1],
        [5, 6, 40, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 3, 48, 1, 1, 1],
        [5, 6, 96, 1, 1, 2],
        [5, 6, 96, 1, 1, 1],
        [5, 6, 96, 1, 1, 1],
    ]

    return MobileNetV3(cfgs, mode='small', **kwargs)
