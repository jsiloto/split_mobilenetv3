import time
import torch
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

def validate(val_loader, val_loader_len, model, criterion, title='Val'):
    num_classes = model.module.classifier[3].out_features
    bar = Bar(title, max=val_loader_len)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    class_prec = [AverageMeter() for i in range(num_classes)]
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)

        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        for i in range(len(target)):
            t = target[i:i+1]
            o = output[i:i+1]
            prec1, prec5 = accuracy(o, t, topk=(1, 5))
            class_prec[t.item()].update(prec1.item(), 1)

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
    top1classes = [c.avg for c in class_prec]

    return (losses.avg, top1.avg, top1classes)
