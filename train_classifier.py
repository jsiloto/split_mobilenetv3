'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
import json
import os
import shutil
import time
import wandb
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from matplotlib import pyplot as plt

from dataset import get_dataset
from eval_classifier import validate
from models.ema import EMA
from models.models import get_model, resume_model, resume_optimizer, resume_training_state
from utils import Bar, Logger, AverageMeter, accuracy, savefig
from compressai_trainer.plot import plot_entropy_bottleneck_distributions

def init_wandb(configs):
    if configs['wandb']:
        wandb.init(
            project="concept_compression",
            config=configs,
            name=configs['project'] + "/" + configs['name'],
        )


class LRAdjust:
    def __init__(self, config):
        self.lr = config['lr']
        self.warmup = config['warmup']
        self.epochs = config['epochs']

    def adjust(self, optimizer, epoch, iteration, num_iter):
        gamma = 0.1
        warmup_epoch = 5 if self.warmup else 0
        warmup_iter = warmup_epoch * num_iter
        current_iter = iteration + epoch * num_iter
        max_iter = self.epochs * num_iter


        if epoch < warmup_epoch:
            lr = self.lr * current_iter / warmup_iter
        else:
            multiplier = (gamma ** ((current_iter - warmup_iter) / (max_iter - warmup_iter)))
            lr = self.lr * multiplier
            print(multiplier)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr




def train_classifier(configs):
    init_wandb(configs)

    d = get_dataset(configs['dataset'], configs['hyper']['batch_size'])
    model = get_model(configs['model']['base_model'], configs['model']['model'], num_classes=d.num_classes)
    # define loss function (criterion) and optimizer
    train_criterion = nn.CrossEntropyLoss().cuda()
    val_criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), configs['hyper']['lr'], weight_decay=1e-4)
    adjuster = LRAdjust(configs['hyper'])

    # optionally resume from a checkpoint
    checkpoint_path = configs['checkpoint']
    model = resume_model(model, checkpoint_path, best=False)
    optimizer = resume_optimizer(optimizer, checkpoint_path, best=False)
    summary = resume_training_state(checkpoint_path, best=False)
    start_epoch = summary['epoch']
    print(f"=> start epoch {start_epoch}")
    resume = (start_epoch != 0)
    logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title="title", resume=resume)
    logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    ########################################################################################
    num_epochs = configs['hyper']['epochs']
    best_discriminator = summary['best_discriminator']

    for epoch in range(start_epoch, num_epochs):
        print('\nEpoch: [%d | %d]' % (epoch + 1, num_epochs))
        train_summary = train(d.train_loader, d.train_loader_len, model, train_criterion, optimizer, adjuster, epoch)
        val_summary = validate(d.val_loader, d.val_loader_len, model, val_criterion)
        summary.update(train_summary)
        summary.update(val_summary)
        lr = optimizer.param_groups[0]['lr']

        # append logger file
        logger.append([epoch, lr, summary['train_loss'],
                       summary['val_loss'], summary['train_top1'],
                       summary['val_top1']])
        summary['epoch'] = epoch + 1
        is_best = best_discriminator > summary['val_discriminator']

        if is_best:
            summary['best_discriminator'] = summary['val_discriminator']
            summary['best_top1'] = summary['val_top1']
            summary['best_top1classes'] = summary['val_top1classes']
            summary['best_bytes'] = summary['val_bytes']
        checkpoint_file = save_checkpoint({
            'summary': summary,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=checkpoint_path)

        summary['step'] = epoch
        if configs['wandb']:
            wandb.log(summary)

    logger.close()
    logger.plot()
    savefig(os.path.join(checkpoint_path, 'log.eps'))

    fig = plot_entropy_bottleneck_distributions(model.encoder.codec.entropy_bottleneck)
    fig.write_image(file='my_figure.png', format='png')

    model = resume_model(model, checkpoint_path, best=True)
    final_summary = summary
    final_summary.update(validate(d.train_loader, d.train_loader_len, model, val_criterion, title='Train Set'))
    final_summary.update(validate(d.val_loader, d.val_loader_len, model, val_criterion, title='Val Set'))
    print('Best accuracy:')
    print(final_summary['val_top1'])
    print("Best Classes Accuracy")
    print(final_summary['val_top1classes'])
    checkpoint_file = save_checkpoint({
        'summary': final_summary,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, True, checkpoint=checkpoint_path)
    if configs['wandb']:
        wandb.save(checkpoint_file)

    print(final_summary)
    with open(os.path.join(checkpoint_path, 'metadata.json'), "w") as f:
        json.dump(final_summary, f)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))
        filepath = os.path.join(checkpoint, 'model_best.pth.tar')

    return filepath


def train(train_loader, train_loader_len, model, criterion, optimizer, adjuster, epoch):
    bar = Bar('Train', max=train_loader_len)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_c = AverageMeter()
    top5meter = AverageMeter()
    top1meter = AverageMeter()

    # switch to train mode
    model.train()


    end = time.time()
    adjuster.adjust(optimizer, epoch, 0, train_loader_len)
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        # a = student.compress(input.to('cuda'))
        # print(len(a['strings'][0][0]))
        # exit()

        data_time.update(time.time() - end)
        target = target.cuda(non_blocking=True)
        output = model(input.to('cuda'))

        compression_loss = output['compression_loss']
        loss = criterion(output['y_hat'], target)
        # measure accuracy and record loss
        with torch.no_grad():
            y_hat = model(input.to('cuda'))['y_hat']
        top1, top5 = accuracy(y_hat, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        losses_c.update(compression_loss.item(), input.size(0))
        top1meter.update(top1.item(), input.size(0))
        top5meter.update(top5.item(), input.size(0))

        # compute gradient and do SGD step
        loss += compression_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = f'({i + 1}/{train_loader_len})' \
                     f'D/B/D+B: {data_time.avg:.2f}s/{batch_time.avg:.2f}s ' \
                     f'| LR: {optimizer.param_groups[0]["lr"]:.4f} |' \
                     f' ETA: {bar.eta_td:} | Loss: {losses.avg:.3f} | CLoss: {losses_c.avg:.3f} |' \
                     f'top1: {top1meter.avg: .2f} | top5: {top5meter.avg: .2f}'
        bar.next()


    bar.finish()

    summary = {
        'train_loss': losses.avg,
        'train_closs': losses_c.avg,
        'train_acc': top1meter.avg,
        'train_top1': top1meter.avg,
        'train_top5': top5meter.avg,
    }

    return summary
