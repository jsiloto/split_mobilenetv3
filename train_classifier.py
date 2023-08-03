'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
import argparse
import json
import os
import random
import shutil
import time
import warnings
import yaml

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from math import cos, pi, ceil
from dataset import get_dataset
from eval_classifier import validate
from mobilenetv3 import mobilenetv3
from model import get_model, resume_model
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig



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
        lr = self.lr * (gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))

        if epoch < warmup_epoch:
            lr = self.lr * current_iter / warmup_iter

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train_classifier(dataset_name, model_config):


    d = get_dataset(dataset_name, model_config['batch_size'])

    model = get_model(model_config, num_classes=d.num_classes)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), model_config['lr'], momentum=0.9, weight_decay=1e-4)
    adjuster = LRAdjust(model_config)

    # optionally resume from a checkpoint
    checkpoint_path = model_config['checkpoint']
    model.codec.entropy_bottleneck.update()
    model, start_epoch, best_prec1 = resume_model(model, checkpoint_path, optimizer, best=False)
    resume = (start_epoch != 0)
    logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title="title", resume=resume)
    logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    ########################################################################################
    metadata = {}
    metadata['dataset_name'] = dataset_name
    metadata['model'] = model_config
    with open(os.path.join(checkpoint_path, 'metadata.json'), "w") as f:
        json.dump(metadata, f)

    for epoch in range(start_epoch, model_config['epochs']):
        print('\nEpoch: [%d | %d]' % (epoch + 1, model_config['epochs']))
        train_loss, train_acc = train(d.train_loader, d.train_loader_len, model, criterion, optimizer, adjuster, epoch)
        val_loss, prec1, top1classes = validate(d.val_loader, d.val_loader_len, model, criterion)
        lr = optimizer.param_groups[0]['lr']

        # append logger file
        logger.append([epoch, lr, train_loss, val_loss, train_acc, prec1])

        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
            best_prec1classes = top1classes
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_prec1classes': best_prec1classes,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=checkpoint_path)

    logger.close()
    logger.plot()
    savefig(os.path.join(checkpoint_path, 'log.eps'))

    model, epoch, best_prec1 = resume_model(model, checkpoint_path, optimizer, best=True)
    validate(d.train_loader, d.train_loader_len, model, criterion, title='Train Set')
    _, prec1, top1classes = validate(d.val_loader, d.val_loader_len, model, criterion, title='Val Set')
    print('Best accuracy:')
    print(prec1)
    print("Classes Accuracy")
    print(top1classes)

    metadata['results'] = {}
    metadata['results']["prec1"] = prec1
    metadata['results']["best_prec1classes"] = top1classes
    metadata['results']["val_samples"] = d.val_samples
    metadata['results']["train_samples"] = d.train_samples

    with open(os.path.join(checkpoint_path, 'metadata.json'), "w") as f:
        json.dump(metadata, f)




def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def train(train_loader, train_loader_len, model, criterion, optimizer, adjuster, epoch):
    bar = Bar('Train', max=train_loader_len)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        adjuster.adjust(optimizer, epoch, i, train_loader_len)

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        # compute output
        output = model(input.to('cuda'))
        y_hat = output['y_hat']
        likelihoods = output['likelihoods']
        lloss = -likelihoods.mean()
        loss = criterion(y_hat, target) + lloss
        # measure accuracy and record loss
        prec1, prec5 = accuracy(y_hat, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | ' \
                     'Batch: {bt:.3f}s | Total: {total:} |' \
                     ' ETA: {eta:} | Loss: {loss:.4f} | ' \
                     'top1: {top1: .4f} | top5: {top5: .4f} | LLos: {lloss: .4f}'.format(
            batch=i + 1,
            size=train_loader_len,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
            lloss=lloss
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)



