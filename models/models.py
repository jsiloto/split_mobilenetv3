import os

import torch
from torch import nn

from models.mobilenetv3.mobilenetv3 import MobileNetV3, mobilenetv3_large
from utils import mkdir_p


def load_mobilenetv3(model_config, num_classes=10):
    # create model
    model = mobilenetv3_large(num_classes=num_classes, **model_config)
    if model_config['pretrained']:
        state_dict = torch.load('models/mobilenetv3/pretrained/mobilenetv3-large-1cd25616.pth')
        state_dict.pop("classifier.3.weight")
        state_dict.pop("classifier.3.bias")
        for k, v in state_dict.items():
            print(k)
        model.load_state_dict(state_dict, strict=False)

    model = model.cuda()
    return model


class MobilenetV3Regular(nn.Module):
    def __init__(self, base_model, **kwargs):
        super(MobilenetV3Regular, self).__init__()
        self.base_model = base_model
        self.num_classes = base_model.classifier[3].out_features

    def forward(self, x):
        output = {'y_hat': self.base_model(x),
                  'strings': None,
                  'likelihoods': None,
                  'num_bytes': 0.0,
                  'reg_loss': 0.0}
        return output


def get_model(base_model_config, model_config, num_classes=10):
    base_model = load_mobilenetv3(base_model_config, num_classes=num_classes)

    model_dict = {
        "regular": MobilenetV3Regular,
    }

    model = model_dict[model_config['name']](**model_config, base_model=base_model)

    return model

def load_checkpoint(checkpoint_path, best=False):
    # if not os.path.isdir(checkpoint_path):
    #     mkdir_p(checkpoint_path)
    ckpt = "checkpoint.pth.tar" if not best else "model_best.pth.tar"
    checkpoint_file = os.path.join(checkpoint_path, ckpt)
    if os.path.isfile(checkpoint_file):
        print(f"=> loading checkpoint {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file)
        return checkpoint
    else:
        return None


def resume_model(model, checkpoint_path, best=False):
    checkpoint = load_checkpoint(checkpoint_path, best)
    if checkpoint is None:
        print(f"=> no checkpoint found at {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint['state_dict'])

    return model


def resume_optimizer(optimizer, checkpoint_path, best=False):
    checkpoint = load_checkpoint(checkpoint_path, best)
    if checkpoint is None:
        print(f"=> no checkpoint found at {checkpoint_path}")
    else:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return optimizer


def resume_training_state(checkpoint_path, best=False):
    metadata = {
        'epoch': 0,
        'best_prec1': 0.0,
        'best_prec1classes': 0.0,
    }

    checkpoint = load_checkpoint(checkpoint_path, best)
    if checkpoint is None:
        print(f"=> no checkpoint found at {checkpoint_path}")
    else:
        metadata = checkpoint['metadata']
    return metadata
