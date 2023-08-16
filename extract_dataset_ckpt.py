import torch

from models.models import get_model, resume_model

configs = {}
configs['base_model'] = {
    'pretrained': True,
    'width_mult': 1.0
}

configs['model'] = {
    'name': 'regular',
}
checkpoint_path='./checkpoints/baseline/stl10_regular_default/model_best.pth.tar'

model = get_model(configs['base_model'], configs['model'], num_classes=10)
model = resume_model(model, checkpoint_path, best=True)

torch.save(model.base_model.state_dict(), 'models/mobilenetv3/pretrained/mobilenetv3-large-stl10.pth')