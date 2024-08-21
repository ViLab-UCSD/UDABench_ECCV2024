import torch
import math
import torch.nn.functional as F

def train_adamatch(batch_iterator, model_fe, model_cls, opt, it, criterion_cls, 
                    cfg, logger, writer):

    # setting training mode
    model_fe.train()
    model_cls.train()
    opt.zero_grad()

    # get data
    ((img_src_weak, img_src_strong), lbl_src), ((img_tgt_weak, img_tgt_strong), _) = next(batch_iterator)
    img_src_weak, img_src_strong, img_tgt_weak, img_tgt_strong, lbl_src = img_src_weak.cuda(), img_src_strong.cuda(), img_tgt_weak.cuda(), img_tgt_strong.cuda(), lbl_src.cuda()

    source_bs = len(img_src_weak) ## source batch size

    ## forward pass the combined mini-batch.
    img_weak = torch.cat([img_src_weak, img_tgt_weak])
    img_strong = torch.cat([img_src_strong, img_tgt_strong])

    output_weak = model_cls(model_fe(img_weak), feat=False)
    output_strong = model_cls(model_fe(img_strong), feat=False)

    ## split outputs into different vectors
    logits_src_weak, logits_tgt_weak = output_weak[:source_bs], output_weak[source_bs:]
    logits_src_strong, logits_tgt_strong = output_strong[:source_bs], output_strong[source_bs:]

    ## switch off gradient propagation to batch norm layers.
    _disable_batchnorm_tracking(model_fe)
    _disable_batchnorm_tracking(model_cls)
    output_source_only = model_cls(model_fe(torch.cat([img_src_weak, img_src_strong])), feat=False)
    source_only_weak = output_source_only[:source_bs]
    source_only_strong = output_source_only[source_bs:]
    _enable_batchnorm_tracking(model_fe)
    _enable_batchnorm_tracking(model_cls)

    ## Part 1: random logit interpolation for weak and strong source samples.
    random_factor = torch.rand_like(source_only_weak)
    interpolated_logit_weak = random_factor * source_only_weak + (1 - random_factor) * logits_src_weak
    random_factor = torch.rand_like(source_only_strong)
    interpolated_logit_strong = random_factor * source_only_strong + (1 - random_factor) * logits_src_strong

    ## supervised loss
    loss_supervised = 0.5*(
        torch.mean(criterion_cls(interpolated_logit_weak, lbl_src)) + torch.mean(criterion_cls(interpolated_logit_strong, lbl_src))
    )

    ## Part 2.1: Distribution Alignment on weakly augmented target images using source samples.
    
    source_weak_softmax = F.softmax(interpolated_logit_weak.detach(), dim=1)
    target_weak_softmax = F.softmax(logits_tgt_weak.detach(), dim=1)

    label_ratio = (1e-6 + source_weak_softmax.mean(0)) / (1e-6 + target_weak_softmax.mean(0))
    unnorm_target_prob = target_weak_softmax * label_ratio
    target_prob = unnorm_target_prob / unnorm_target_prob.sum(1, keepdims=True)

    ## Part 2.2: Relative Confidence Thresholding
    mean_source_conf = source_weak_softmax.max(1)[0].mean(0)
    c_tau = cfg["training"]["tau"] * mean_source_conf
    target_mask = target_prob.max(1)[0] > c_tau
    target_pseudo_labels = target_prob.argmax(1)

    loss_target = (criterion_cls(logits_tgt_strong, target_pseudo_labels.detach()) * target_mask.detach()).mean()
    coeff = _compute_loss_target_weight(it, cfg['training']['iteration'])

    loss = loss_supervised + coeff*loss_target

    # back propagation
    loss.backward()
    opt.step()

    curr_lr = opt.param_groups[0]['lr']
    if (it + 1) % cfg['training']['print_interval'] == 0:
        print_str = 'Iteration: [{0}/{1}]\t' \
            'LR: [{curr_lr:.4g}]\t' \
            'CLoss {closs:.4f}\t' \
            'TgtLoss {tgtloss:.4f}'.format(
                it + 1, cfg['training']['iteration'], curr_lr=curr_lr,
                closs=loss_supervised.item(), tgtloss=loss_target.item()
            )

        logger.info(print_str)

    writer.add_scalar('train/lr', curr_lr, it + 1)
    writer.add_scalar('train/c_loss', loss_supervised.item(), it + 1)
    writer.add_scalar('train/tgt_loss', loss_target.item(), it + 1)


def _disable_batchnorm_tracking(model):
    def fn(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False

    model.apply(fn)

def _enable_batchnorm_tracking(model):
    def fn(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = True

    model.apply(fn)

def _compute_loss_target_weight(n_iter, max_iter):
    mu = 0.5 - math.cos(min(math.pi, 2 * math.pi * n_iter / max_iter)) / 2

    return mu
