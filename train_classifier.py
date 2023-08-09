'''
Training script for ImageNet
Copyright (c) Wei YANG, 2017
'''
import json
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from dataset import get_dataset
from eval_classifier import validate
from models.models import get_model, resume_model, resume_optimizer, resume_training_state
from utils import Bar, Logger, AverageMeter, accuracy, savefig



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


def train_classifier(configs):
    d = get_dataset(configs['dataset'], configs['hyper']['batch_size'])
    model = get_model(configs['base_model'], configs['model'], num_classes=d.num_classes)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), configs['hyper']['lr'], momentum=0.9, weight_decay=1e-4)
    adjuster = LRAdjust(configs['hyper'])

    # optionally resume from a checkpoint
    checkpoint_path = configs['checkpoint']
    model = resume_model(model, checkpoint_path, best=False)
    optimizer = resume_optimizer(optimizer, checkpoint_path, best=False)
    training_state = resume_training_state(checkpoint_path, best=False)
    start_epoch = training_state['epoch']
    resume = (start_epoch != 0)
    logger = Logger(os.path.join(checkpoint_path, 'log.txt'), title="title", resume=resume)
    logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    ########################################################################################

    num_epochs = configs['hyper']['epochs']
    for epoch in range(start_epoch, num_epochs):
        print('\nEpoch: [%d | %d]' % (epoch + 1, num_epochs))
        train_loss, train_acc = train(d.train_loader, d.train_loader_len, model, criterion, optimizer, adjuster, epoch)
        val_loss, prec1, top1classes = validate(d.val_loader, d.val_loader_len, model, criterion)
        lr = optimizer.param_groups[0]['lr']

        # append logger file
        logger.append([epoch, lr, train_loss, val_loss, train_acc, prec1])

        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = prec1
            best_prec1classes = top1classes

        metadata = {
            'epoch': epoch + 1,
            'best_prec1': best_prec1,
            'best_prec1classes': best_prec1classes,
        }

        save_checkpoint({
            'metadata': metadata,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=checkpoint_path)

    logger.close()
    logger.plot()
    savefig(os.path.join(checkpoint_path, 'log.eps'))

    model = resume_model(model, checkpoint_path, best=True)
    validate(d.train_loader, d.train_loader_len, model, criterion, title='Train Set')
    _, prec1, top1classes = validate(d.val_loader, d.val_loader_len, model, criterion, title='Val Set')
    print('Best accuracy:')
    print(prec1)
    print("Classes Accuracy")
    print(top1classes)

    results = {}
    results["prec1"] = prec1
    results["best_prec1classes"] = top1classes
    results["val_samples"] = d.val_samples
    results["train_samples"] = d.train_samples

    with open(os.path.join(checkpoint_path, 'metadata.json'), "w") as f:
        json.dump(results, f)




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
        likelihoods = output['likelihoods']['y'].log2()
        lloss = -likelihoods.mean()
        loss = criterion(y_hat, target)
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
        bar.suffix = f'({i + 1}/{train_loader_len}) Data: {data_time.avg:.3f}s | ' \
                     f'Batch: {batch_time.avg:.3f}s | Total: {bar.elapsed_td:} |' \
                     f' ETA: {bar.eta_td:} | Loss: {losses.avg:.4f} | ' \
                     f'top1: {top1.avg: .4f} | top5: {top5.avg: .4f} | LLos: {lloss: .4f}'
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)