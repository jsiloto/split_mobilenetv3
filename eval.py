import argparse
import json
import os
import shutil

import numpy as np
import torch
import yaml
from torch import nn

from configs import get_config_from_args
from dataset import get_dataset
from distill_classifier import distill_classifier
from eval_classifier import validate
from models.models import get_model, resume_model, resume_training_state
from train_classifier import train_classifier
from utils import mkdir_p

def main():
    parser = argparse.ArgumentParser(description='Train Model')
    configs = get_config_from_args(parser)
    with open(os.path.join(configs['checkpoint'], 'metadata.json'), "w") as f:
        json.dump(configs, f)

    print(configs)

    d = get_dataset(configs['dataset'], configs['hyper']['batch_size'])
    model = get_model(configs['model']['base_model'], configs['model']['model'], num_classes=d.num_classes)
    # define loss function (criterion) and optimizer
    val_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), configs['hyper']['lr'], weight_decay=1e-4)

    # optionally resume from a checkpoint
    checkpoint_path = configs['checkpoint']
    model = resume_model(model, checkpoint_path, best=False)
    summary = resume_training_state(checkpoint_path, best=False)
    print("Resuming from epoch", summary['epoch'])

    # for tier in range(len(model.encoder.betas)):
    top_summary = {}
    for tier in np.linspace(0, len(model.encoder.betas)-1, len(model.encoder.betas)*3):
        val_summary = validate(d.val_loader, d.val_loader_len, model, val_criterion, tier=tier)
        top_summary[tier] = val_summary

    with open(os.path.join(configs['checkpoint'], 'eval_summary.json'), "w") as f:
        json.dump(top_summary, f)


if __name__ == '__main__':
    main()
