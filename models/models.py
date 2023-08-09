from models.mobilenetv3.mobilenetv3 import MobileNetV3, mobilenetv3_large


def load_mobilenetv3(model_config, num_classes=10):
    # create model
    model_params = model_config['model_params']
    model = mobilenetv3.mobilenetv3_large(num_classes=num_classes, width_mult=1.0,
                                          **model_params)
    if model_config['pretrained']:
        state_dict = torch.load('mobilenetv3/pretrained/mobilenetv3-large-1cd25616.pth')
        state_dict.pop("classifier.3.weight")
        state_dict.pop("classifier.3.bias")
        model.load_state_dict(state_dict, strict=False)

    model = model.cuda()
    return model


def get_model(base_model_config, model_config, num_classes=10):

    base_model = mobilenetv3_large(num_classes=num_classes, **base_model_config)


    model_dict = {
        "regular": MobileNetV3,
    }

    model = model_dict[model_config['name']](**model_config, base_model=base_model)

    return model


def get_model(model_config, num_classes=10):
    # create model
    model_params = model_config['model_params']
    model = mobilenetv3.mobilenetv3_large(num_classes=num_classes, width_mult=1.0,
                                          **model_params)
    if model_config['pretrained']:
        state_dict = torch.load('models/mobilenetv3/pretrained/mobilenetv3-large-1cd25616.pth')
        state_dict.pop("classifier.3.weight")
        state_dict.pop("classifier.3.bias")
        model.load_state_dict(state_dict, strict=False)

    model = model.cuda()
    return model

def resume_model(model, checkpoint_path, optimizer=None, best=False):
    best_prec1 = 0.0
    if not os.path.isdir(checkpoint_path):
        mkdir_p(checkpoint_path)

    ckpt = "checkpoint.pth.tar" if not best else "model_best.pth.tar"
    checkpoint_file = os.path.join(checkpoint_path, ckpt)
    if os.path.isfile(checkpoint_file):
        print(f"=> loading checkpoint {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file)
        epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"=> loaded checkpoint {checkpoint_file} (epoch {epoch})")


    else:
        epoch = 0
        print(f"=> no checkpoint found at {checkpoint_file}")


    return model, epoch, best_prec1
