import torch
from metrics import averageMeter, accuracy

def val(data_loader, model_fe, model_cls, it, n_classes, domain, logger, writer):

    # setup average meters
    losses = averageMeter()
    top1 = averageMeter()
    top5 = averageMeter()

    # setting training mode
    model_fe.eval()
    model_cls.eval()

    all_preds = []
    all_labels = []
    cls_num_list_tgt = []
    len_dl = len(data_loader)
    print()
    for (step, value) in enumerate(data_loader):

        image = value[0].cuda()
        target = value[1].cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if isinstance(model_cls, (list, tuple)):
                features = model_fe(image)
                output = torch.sum(torch.stack([cls(features) for cls in model_cls]), dim=0)
            else:
                output = model_cls(model_fe(image), feat=False)

        # measure accuracy
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1, image.size(0))
        top5.update(prec5, image.size(0))

        # per class accuracy metrics
        all_preds.extend(output.argmax(1).cpu().numpy().tolist())
        all_labels.extend(target.cpu().numpy().tolist())

    logger.info('[Val] Iteration {it}\tTop 1 Acc {top1.avg:.3f}\tTop 5 Acc. {top5.avg:.3f}'.format(it=it+1, top1=top1, top5=top5))

    writer.add_scalar('val/loss_{}'.format(domain), losses.avg, it + 1)
    writer.add_scalar('val/top1_{}'.format(domain), top1.avg, it + 1)
    writer.add_scalar('val/top5_{}'.format(domain), top5.avg, it + 1)

    # setting training mode
    model_fe.train()
    model_cls.train()

    return top1.avg, top5.avg