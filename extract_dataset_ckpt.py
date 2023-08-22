import torch

from models.models import get_model, resume_model

configs = {}
configs['base_model'] = {
    'pretrained': True,
    'width_mult': 1.0
}

configs['model'] = {
        'name': 'channel_bottleneck',
        'split_position': 5,
        'bottleneck_ratio': 1.0,
        'checkpoint': "checkpoints/baseline/stl10_channel_bottleneck_1.0_default/model_best.pth.tar"
}

model = get_model(configs['base_model'], configs['model'], num_classes=10)

torch.save(model.base_model.state_dict(), 'models/mobilenetv3/pretrained/mobilenetv3-large-stl10.pth')