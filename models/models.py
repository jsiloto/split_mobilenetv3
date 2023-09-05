import os

import torch
from torch import nn

from models.lic.gain_compressor import GainCompressor
from models.mobilenetv3.mobilenetv3 import MobileNetV3, mobilenetv3_large
from models.split.channel_bottleneck import MV3ChannelBottleneck
from models.split.entropy_bottleneck import MV3EntropyBottleneck
from models.split.entropy_bottleneck2 import MV3EntropyBottleneck2
from models.split.entropy_precompressor import MV3Precompressor
from models.split.gain_bottleneck import MV3GainBottleneck
from models.split.regular import MobilenetV3Regular
from utils import mkdir_p


def load_mobilenetv3(model_config, num_classes=10):
    # create model
    model = mobilenetv3_large(num_classes=num_classes, **model_config)
    checkpoint_path = 'models/mobilenetv3/pretrained/mobilenetv3-large-1cd25616.pth'
    if 'checkpoint' in model_config:
        checkpoint_path = model_config['checkpoint']

    if model_config['pretrained']:
        state_dict = torch.load(checkpoint_path)
        if state_dict["classifier.3.weight"].shape[0] != num_classes:
            print("Original Weights use different classes, discarding last layer")
            state_dict.pop("classifier.3.weight")
            state_dict.pop("classifier.3.bias")
        model.load_state_dict(state_dict, strict=False)

    model = model.cuda()
    return model



#################################### CHeckpointing ####################################
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
        print(f"=> no model checkpoint found at {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint['state_dict'])
        if hasattr(model.encoder, 'codec'):
            model.encoder.codec.entropy_bottleneck.update()

    return model


def resume_optimizer(optimizer, checkpoint_path, best=False):
    checkpoint = load_checkpoint(checkpoint_path, best)
    if checkpoint is None:
        print(f"=> no optimizer checkpoint found at {checkpoint_path}")
    else:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return optimizer


def resume_training_state(checkpoint_path, best=False):
    summary = {
        'epoch': 0,
        'best_discriminator': 100000,
        'best_bytes': 100000,
        'best_top1': 0.0,
        'best_top1classes': 0.0,
    }

    checkpoint = load_checkpoint(checkpoint_path, best)
    if checkpoint is None:
        print(f"=> no summary checkpoint found at {checkpoint_path}")
    else:
        summary = checkpoint['summary']
    return summary


#################################### Model Dicts ####################################
def get_model(base_model_config, model_config, num_classes=10):
    base_model = load_mobilenetv3(base_model_config, num_classes=num_classes)

    model_dict = {
        "regular": MobilenetV3Regular,
        "channel_bottleneck": MV3ChannelBottleneck,
        "entropy_bottleneck": MV3EntropyBottleneck,
        "entropy_bottleneck2": MV3EntropyBottleneck2,
        "gain_bottleneck": MV3GainBottleneck,
        "entropy_precompressor": MV3Precompressor,
        "gain_compressor": GainCompressor,
    }

    model = model_dict[model_config['name']](**model_config, base_model=base_model).to('cuda')
    if "checkpoint" in model_config:
        model.load_state_dict(torch.load(model_config["checkpoint"])['state_dict'])

    return model
