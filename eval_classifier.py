import time
import torch
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


def validate(val_loader, val_loader_len, model, criterion, title='Val'):
    num_classes = model.num_classes
    bar = Bar(title, max=val_loader_len)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1meter = AverageMeter()
    top5meter = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    class_prec = [AverageMeter() for i in range(num_classes)]
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(input.to('cuda'))
            y_hat = output['y_hat']
            loss = criterion(y_hat, target)

            # measure accuracy and record loss
            for i in range(len(target)):
                t = target[i:i + 1]
                o = y_hat[i:i + 1]
                top1, top5 = accuracy(o, t, topk=(1, 5))
                class_prec[t.item()].update(top1.item(), 1)

            top1, top5 = accuracy(y_hat, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1meter.update(top1.item(), input.size(0))
            top5meter.update(top5.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = f'({i + 1}/{val_loader_len}) ' \
                         f'D/B/D+B: {data_time.avg:.2f}s/{batch_time.avg:.2f}s | T: {bar.elapsed_td:}' \
                         f' | ETA: {bar.eta_td:} | Loss: {losses.avg:.2f} | top1: {top1meter.avg: .2f}' \
                         f' | top5: {top5meter.avg: .2f} | Bytes: ----'
            bar.next()

        # Record Bytes at the end of the epoch to reduce computational overhead
        output = model.compress(input.to('cuda'))
        num_bytes = output['num_bytes']
        bar.suffix = f'({i + 1}/{val_loader_len}) ' \
                     f'D/B/D+B: {data_time.avg:.2f}s/{batch_time.avg:.2f}s | T: {bar.elapsed_td:}' \
                     f' | ETA: {bar.eta_td:} | Loss: {losses.avg:.2f} | top1: {top1meter.avg: .2f}' \
                     f' | top5: {top5meter.avg: .2f} | Bytes: {num_bytes: .1f}'
        bar.next()

    bar.finish()
    top1classes = [c.avg for c in class_prec]

    summary = {
        'val_top1': top1meter.avg,
        'val_top5': top5meter.avg,
        'val_bytes': num_bytes,
        'val_top1classes': top1classes,
        'val_loss': losses.avg,
    }

    return summary
