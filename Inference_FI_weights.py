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
from mobilenetv3 import mobilenetv3
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter

import pytorchfi
from pytorchfi import core
from pytorchfi import neuron_error_models
from pytorchfi import weight_error_models

from pytorchfi.core import FaultInjection

from pytorchfi.FI_Weights import FI_report_classifier
from pytorchfi.FI_Weights import FI_framework
from pytorchfi.FI_Weights import FI_manager 
from pytorchfi.FI_Weights import DatasetSampling 

# comment this line, otherwise the fault injections will collapse due to leaking memory produced by 'file_system
#torch.multiprocessing.set_sharing_strategy('file_system')
import logging
from logging import FileHandler, Formatter

import os
import pickle
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--config', required=True, help='yaml file path')
parser.add_argument('-d', '--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--lr-decay', type=str, default='step',
                    help='mode for learning rate decay')
parser.add_argument('--step', type=int, default=30,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')
parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')

parser.add_argument('--weight', default='', type=str, metavar='WEIGHT',
                    help='path to pretrained weight (default: none)')

best_prec1 = 0



LOGGING_FORMAT = '%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s'

logging.basicConfig(
    format=LOGGING_FORMAT,
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.INFO,
)
def_logger = logging.getLogger()

logger = def_logger.getChild(__name__)


def make_parent_dirs(file_path):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

def setup_log_file(log_file_path):
    make_parent_dirs(log_file_path)
    fh = FileHandler(filename=log_file_path, mode='w')
    fh.setFormatter(Formatter(LOGGING_FORMAT))
    def_logger.addHandler(fh)

def get_transforms(split='train', input_size=(128, 128)):

    if split == 'train':
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize(input_size),
            transforms.RandomRotation(degrees=(-20,20)),
            # transforms.ColorJitter(brightness=.4, hue=.25),
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
        ])
    elif split == 'test':
        transform = transforms.Compose([
            # you can add other transformations in this list
            # transforms.Resize(input_size),
            transforms.ToTensor()
        ])
    return transform


def main():
    global args, best_prec1
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.full_load(f)

    setup_log_file(os.path.expanduser("log/log.log"))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'Tbarhis will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    print(config)
    split_position = config.get('split_position', -1)
    bottleneck_channels = config.get('bottleneck_channels', -1)
    model = mobilenetv3.SplitMobileNetV3(num_classes=10, pretrained=args.pretrained, split_position=split_position, bottleneck_channels=bottleneck_channels)

    if not args.distributed:
        model = torch.nn.DataParallel(model).cuda()
    else:
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    title = 'test'
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.checkpoint = os.path.dirname(args.resume)
            logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(['Epoch', 'Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

    ############################ Datasets ########################################
    cudnn.enabled=True
    # cudnn.benchmark = True
    cudnn.deterministic = True
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    # in PyTorch 1.12 and later.
    torch.backends.cuda.matmul.allow_tf32 = True
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    cudnn.allow_tf32 = True

    train_dataset = datasets.STL10(root="./data/stl10", transform=get_transforms(split='train'),
                                   download=True, split="train")
    val_dataset = datasets.STL10(root="./data/stl10", transform=get_transforms(split='test'),
                                 download=True, split="test")


    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=args.workers)

    train_loader_len = ceil(len(train_dataset)/args.batch_size)
    val_loader_len = ceil(len(val_dataset)/args.batch_size)



    name_config=((args.config.split('/'))[-1]).replace(".yaml","")
    conf_fault_dict=config['fault_info']['weights']
    name_config=f"FSIM_logs/{name_config}_weights_{conf_fault_dict['layer'][0]}"

    cwd=os.getcwd() 
    model.eval() 
    print(model)
    # student_model.deactivate_analysis()
    full_log_path=os.path.join(cwd,name_config)
    # 1. create the fault injection setup
    FI_setup=FI_manager(full_log_path,"ckpt_FI.json","fsim_report.csv")

    # 2. Run a fault free scenario to generate the golden model
    FI_setup.open_golden_results("Golden_results")
    validate(val_loader, val_loader_len, model, criterion, fsim_enabled=True, Fsim_setup=FI_setup)
    FI_setup.close_golden_results()


    
    # 3. Prepare the Model for fault injections
    FI_setup.FI_framework.create_fault_injection_model(torch.device('cuda'),model,
                                        batch_size=1,
                                        input_shape=[3,96,96],
                                        layer_types=[torch.nn.Conv2d,torch.nn.Linear])
    
    # 4. generate the fault list
    logging.getLogger('pytorchfi').disabled = True
    FI_setup.generate_fault_list(flist_mode='sbfm',f_list_file='fault_list.csv',layer=conf_fault_dict['layer'][0])    
    FI_setup.load_check_point()

    # 5. Execute the fault injection campaign
    for fault,k in FI_setup.iter_fault_list():
        # 5.1 inject the fault in the model
        FI_setup.FI_framework.bit_flip_weight_inj(fault)
        FI_setup.open_faulty_results(f"F_{k}_results")
        try:   
            # 5.2 run the inference with the faulty model 
            validate(val_loader, val_loader_len, FI_setup.FI_framework.faulty_model, criterion, fsim_enabled=True, Fsim_setup=FI_setup)
        except Exception as Error:
            msg=f"Exception error: {Error}"
            logger.info(msg)
        # 5.3 Report the results of the fault injection campaign
        FI_setup.close_faulty_results()
        FI_setup.parse_results()
        FI_setup.write_reports()

    ########################################################################################

    # if args.evaluate:
    #     validate(train_loader, train_loader_len, model, criterion)
    #     validate(val_loader, val_loader_len, model, criterion)
    #     return

    # visualization
    # writer = SummaryWriter(os.path.join(args.checkpoint, 'logs'))

   


def validate(val_loader, val_loader_len, model, criterion, fsim_enabled=False, Fsim_setup:FI_manager = None):
    bar = Bar('Val', max=val_loader_len)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        #logger.info(f"Input={input.shape}")
        with torch.no_grad():
            # compute output            
            output = model(input)
            loss = criterion(output, target)
            if fsim_enabled==True:
                Fsim_setup.FI_report.update_report(i,output,target,topk=(1,5))

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .3f} | top5: {top5: .3f}'.format(
            batch=i + 1,
            size=val_loader_len,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


from math import cos, pi, ceil


def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if args.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
