import numpy as np
import torch
import torchvision
from torch import nn

from utils.jpeg import JPEGCompression


class MobilenetV3Regular(nn.Module):
    def __init__(self, base_model, **kwargs):
        super(MobilenetV3Regular, self).__init__()
        self.base_model = base_model
        self.num_classes = base_model.classifier[3].out_features

    def forward(self, x):
        output = {'y_hat': self.base_model(x),
                  'strings': None,
                  'likelihoods': None,
                  'compression_loss': torch.tensor(0.0)}
        return output

    def compress(self, x):
        pixels = x.shape[-1] * x.shape[-2] * x.shape[-3]
        jpeg = JPEGCompression(quality=95)
        for xx in x:
            xx = torchvision.transforms.functional.to_pil_image(xx)
            jpeg(xx)

        output = {'num_bytes': jpeg.average_size(),
                  'bpp': jpeg.average_size() / pixels}
        return output