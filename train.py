import argparse
import os
import yaml
import random
import shutil
import numpy as np
import torch
from torch import inverse, nn
import torch.nn.functional as F

from loader import get_dataloader
from models import get_model
from optimizers import get_optimizer, get_scheduler
from UDA_trainer import get_trainer, val
from losses import get_loss
from utils import cvt2normal_state, get_logger, loop_iterable, get_parameter_count, disable_batchnorm_tracking

torch.autograd.set_detect_anomaly(True)

from tensorboardX import SummaryWriter

def main():

    if not torch.cuda.is_available():
        raise SystemExit('GPU is needed')

    # setup random seeds
    seed=cfg.get('seed', 1234)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # setup data loader
    splits = ['train', 'val']
    data_loader_src = get_dataloader(cfg['data']['source'], splits, cfg['training']['batch_size'])
    if cfg['training']['trainer'] == "adamatch":
        cfg['training']['batch_size'] *= 3 #cfg["training"]["target_batch_multipler"]
    data_loader_tgt = get_dataloader(cfg['data']['target'], splits, cfg['training']['batch_size'])
    batch_iterator = zip(loop_iterable(data_loader_src['train']), loop_iterable(data_loader_tgt['train']))

    cls_num_list = data_loader_src['cls_num_list']
    n_classes = cfg["data"]["target"]["n_class"]
    cls_num_list_tgt = [0 for _ in range(n_classes)]

    # setup model (feature extractor(s) + classifier(s) + discriminator)
    n_gpu = torch.cuda.device_count()
    model_fe = get_model(cfg['model']['feature_extractor']).cuda()
    # logger.info(model_fe)
    params = [{'params': model_fe.parameters(), 'lr': 1}]
    fe_list = [model_fe]

    if cfg['model'].get('feature_extractor_2', None):
        param_dict = cfg['model']['feature_extractor_2']
        trainable = param_dict['trainable']
        param_dict.pop("trainable")
        model_fe_2 = get_model(param_dict).cuda()
        if trainable:
            params += [{'params': model_fe_2.parameters(), 'lr': 1}]
        fe_list += [model_fe_2]

    model_cls = get_model(cfg['model']['classifier']).cuda()
    params += [{'params': model_cls.parameters(), 'lr': 10}]
    cls_list = [model_cls]

    total_n_params = sum([p.numel() for p in model_fe.parameters()]) + \
                            sum([p.numel() for p in model_cls.parameters()])

    if cfg['model'].get('classifier_2', None):
        model_cls_2 = get_model(cfg['model']['classifier_2']).cuda()
        if cfg['model']['classifier_2']['trainable']:
            params += [{'params': model_cls_2.parameters(), 'lr': 10}]
        cls_list += [model_cls_2]


    d_list = []
    if cfg['model'].get('discriminator', None):
        model_d = get_model(cfg['model']['discriminator']).cuda()
        params += [{'params': model_d.parameters(), 'lr': 10}]
        d_list = [model_d]
    
    # setup loss criterion. Order and names should match in the trainer file and config file.
    loss_dict = cfg['training']['losses']
    criterion_list = []
    for loss_name, loss_params in loss_dict.items():
        criterion_list.append(get_loss(loss_params))

    # setup optimizer
    opt_main_cls, opt_main_params = get_optimizer(cfg['training']['optimizer'])
    opt = opt_main_cls(params, **opt_main_params)

    # setup scheduler
    scheduler = get_scheduler(opt, cfg['training']['scheduler'])
    trainer = get_trainer(cfg["training"])
    
    # if checkpoint already present, resume from checkpoint.
    resume_from_ckpt = False
    if os.path.exists(os.path.join(logdir, 'checkpoint.pkl')):
        cfg['training']['resume']['model'] = os.path.join(logdir, 'checkpoint.pkl')
        cfg['training']['resume']['param_only'] = False
        cfg['training']['resume']['load_cls'] = True
        resume_from_ckpt = True

    # load checkpoint
    start_it = 0
    best_acc_tgt = best_acc_src = 0
    best_acc_tgt_top5 = best_acc_src_top5 = 0
    
    if cfg['training']['resume'].get('model', None):
        resume = cfg['training']['resume']
        resume_model = resume['model']
        if os.path.isfile(resume_model):

            checkpoint = torch.load(resume_model)

            if resume_from_ckpt:
                load_dict = checkpoint["model_fe_state"]
            elif any(substring in resume_model for substring in ["mae", "simclr", "moco", "swav", "sup"]):
                if "mae" in resume_model:
                    load_dict = checkpoint["model"]
                    load_dict = {k:v for k,v in load_dict.items() if not k.startswith("decoder")}
                if "swav" in resume_model or "sup" in resume_model:
                    load_dict = checkpoint["state_dict"]
                    load_dict = {k.partition("module.")[-1]:v for k,v in load_dict.items()}
                if "moco" in resume_model:
                    load_dict = checkpoint["state_dict"]
                    load_dict = {k.partition("base_encoder.")[-1]:v for k,v in load_dict.items()}
                if "simclr" in resume_model:
                    load_dict = checkpoint["model"]
                    load_dict = {k.partition("backbone.")[-1]:v for k,v in load_dict.items()}
                if "mask_token" in load_dict:
                    load_dict.pop("mask_token")
            try:
                ks = model_fe.load_state_dict(load_dict, strict=False)
            except:
                ks = model_fe.load_state_dict(cvt2normal_state(load_dict), strict=False)
            print(ks)
            logger.info('Loading model from checkpoint {}'.format(resume_model))
            ## TODO: add loading additional feature extractors and classifiers
            if resume.get('load_cls', True):
                try:
                    model_cls.load_state_dict((checkpoint['model_cls_state']))
                    logger.info('Loading classifier')
                except:
                    model_cls.load_state_dict(cvt2normal_state(checkpoint['model_cls_state']))
                    logger.info('Loading classifier')
            
            if checkpoint.get('model_d_state', None):
                model_d.load_state_dict((checkpoint['model_d_state']))

            if resume['param_only'] is False:
                start_it = checkpoint['iteration']
                best_acc_tgt = checkpoint.get('best_acc_tgt', 0)
                best_acc_src = checkpoint.get('best_acc_src', 0)
                opt.load_state_dict(checkpoint['opt_main_state'])
                scheduler.load_state_dict(checkpoint['scheduler_state'])
                logger.info('Resuming training state ... ')

            logger.info("Loaded checkpoint '{}'".format(resume_model))
        else:
            logger.info("No checkpoint found at '{}'".format(resume_model))

    logger.info('Start training from iteration {}'.format(start_it))

    if not cfg["training"]["use_bn"]:
        disable_batchnorm_tracking(model_fe)
        disable_batchnorm_tracking(model_cls)

    if n_gpu > 1:
        logger.info("Using multiple GPUs")
        # fe_list = [nn.DataParallel(mfe, device_ids=range(n_gpu)) for mfe in fe_list]
        # cls_list = [nn.DataParallel(mcls, device_ids=range(n_gpu)) for mcls in cls_list]
        model_fe = nn.DataParallel(model_fe, device_ids=range(n_gpu))
        model_cls = nn.DataParallel(model_cls, device_ids=range(n_gpu))

    for it in range(start_it, cfg['training']['iteration']):

        scheduler.step()

        ## Trainer can take multiple feature extractors (eg: FixMatch), and multiple classifiers (eg: MCD)
        trainer(batch_iterator, model_fe, model_cls, *d_list, opt, it, *criterion_list,
                    cfg, logger, writer)

        if (it + 1) % cfg['training']['val_interval'] == 0:
                
            with torch.no_grad():
                acc_src, acc_src_top5 = val(data_loader_src['val'], model_fe, model_cls, it, n_classes, "source", logger, writer)
                acc_tgt, acc_tgt_top5 = val(data_loader_tgt['val'], model_fe, model_cls, it, n_classes, "target", logger, writer)
                is_best = False
                if acc_tgt > best_acc_tgt:
                    is_best = True
                    best_acc_tgt = acc_tgt
                    best_acc_src = acc_src
                    best_acc_tgt_top5 = acc_tgt_top5
                    best_acc_src_top5 = acc_src_top5
                    with open(os.path.join(logdir, 'best_acc.txt'), "a") as fh:
                        write_str = "Source Top 1\t{src_top1:.3f}\tSource Top 5\t{src_top5:.3f}\tTarget Top 1\t{tgt_top1:.3f}\tTarget Top 5\t{tgt_top5:.3f}\n".format(src_top1=best_acc_src, src_top5=best_acc_src_top5, tgt_top1=best_acc_tgt, tgt_top5=best_acc_tgt_top5)
                        fh.write(write_str)
                # if acc_src > best_acc_src:
                #     best_acc_src = acc_src
                print_str = '[Val] Iteration {it}\tBest Acc source. {acc_src:.3f}\tBest Acc target. {acc_tgt:.3f}'.format(it=it+1, acc_src=best_acc_src, acc_tgt=best_acc_tgt)
                logger.info(print_str)

        # if (it + 1) % cfg['training']['save_interval'] == 0:
            ## TODO: add saving additional feature extractors and classifiers
            state = {
                'iteration': it + 1,
                'model_fe_state': model_fe.state_dict(),
                'model_cls_state': model_cls.state_dict(),
                'opt_main_state': opt.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_acc_tgt' : best_acc_tgt,
                'best_acc_src' : best_acc_src
            }

            if len(d_list):
                state['model_d_state'] = model_d.state_dict()
            
            ckpt_path = os.path.join(logdir, 'checkpoint.pkl')
            save_path = ckpt_path#.format(it=it+1)
            torch.save(state, save_path)

            if is_best:
                best_path = os.path.join(logdir, 'best_model.pkl')
                torch.save(state, best_path)
            logger.info('[Checkpoint]: {} saved'.format(save_path))



if __name__ == '__main__':
    global cfg, args, writer, logger, logdir
    valid_trainers = ["plain", "dann", "cdan", "safn", "mcc", "memsac", "mdd", "mcd", "toalign", "hda", "adamatch", "daln", "ilada", "bsp"]
    pt_dataset = ["imnet", "places", "inat", "imnetpre", "none"]
    pt_model = ["mae","simclr","moco","swav","sup","none"]
    backbone_choices = ["resnet50", "resnet101", "vits16", "vitb16","vitl16" , "timm_resnet50", "timm_swin", "timm_convnext", "timm_resmlp", "timm_deit"]

    parser = argparse.ArgumentParser(description='config')
    parser.add_argument(
        '--config',
        nargs='?',
        type=str,
        default='configs/default.yml',
        help='Configuration file to use',
    )
    parser.add_argument("--source" , help="Source file path")
    parser.add_argument("--seed", type=int, default=1234, help="Fix random seed")
    parser.add_argument("--source_test" , help="Source file path for test")
    parser.add_argument("--target_test" , help="Target file path for test")
    parser.add_argument("--target" , help="Target file path")
    parser.add_argument("--lr_rate" , help="Learning Rate", default=0.003, type=float)
    parser.add_argument("--num_class", type=int, help="Number of classes")
    parser.add_argument("--data_root", type=str, help="Data root")
    parser.add_argument("--cbs_source", type=int, default=0, choices=[0,1], help="Class balancing in source")
    parser.add_argument("--source_imb_factor", type=float, default=100, help="Imbalance factor for source. 1 if not resampling. Negative for reverse imbalance.")
    parser.add_argument("--target_imb_factor", type=float, default=100, help="Imbalance factor for target. 1 if not resampling. Negative for reverse imbalance.")
    parser.add_argument("--prune_mode", type=str, default="prune", help="Method to prune the dataset.")
    parser.add_argument("--trainer", required=True, type=str.lower, choices=valid_trainers, help="Adaptation method.")
    parser.add_argument("--norm", type=int, default=0, help="Normalize features [0/1]")
    parser.add_argument("--num_iter", type=int, default=100004, help="Total number of iterations")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--resume", help="Resume training from checkpoint")
    parser.add_argument("--exp_name", help="experiment name")
    parser.add_argument("--pt_model", help="pretrained model name", choices=pt_model)
    parser.add_argument("--pt_dataset", help="pretrained dataset name", choices=pt_dataset)
    parser.add_argument("--backbone", help="backbone network", choices=backbone_choices, default="resnet50")
    parser.add_argument("--linear", help="Linear Probing of SSL Models", action="store_true")
    parser.add_argument("--pass_target", help="Pass the target samples to influence batch norm", type=int)
    parser.add_argument("--use_bn", help="Use batchnorm layers for updates", type=int, default=1)
    parser.add_argument("--val_freq", help="Validation Frequency", type=int, default=5000)


    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    ## Seed
    cfg["seed"] = args.seed

    ## overwrite config parameters
    n_class = args.num_class
    cfg["model"]["classifier"]["n_class"] = n_class
    if args.norm:
        cfg["model"]["classifier"]["norm"] = args.norm
    cfg["data"]["source"]["n_class"] = n_class
    cfg["data"]["target"]["n_class"] = n_class

    cfg["training"]["val_interval"] = args.val_freq

    if args.trainer == "adamatch":
        cfg["data"]["source"]["dual_aug"] = True
        cfg["data"]["target"]["dual_aug"] = True
    else:
        cfg["data"]["source"]["dual_aug"] = False
        cfg["data"]["target"]["dual_aug"] = False

    cfg['pt_model'] = args.pt_model
    if args.resume:
        cfg["training"]["resume"]["model"] = args.resume

    if args.pt_model is not None and args.pt_dataset is not None:
        pt_path = os.path.join("pt_models", args.pt_model + "_" + args.backbone + "_" + args.pt_dataset) + ".pth"
        if os.path.exists(pt_path):
            cfg["training"]["resume"]["model"] = pt_path
    else:
        cfg["training"]["resume"]["model"] = None

    cfg["training"]["pass_target"] = bool(args.pass_target)
    cfg["training"]["use_bn"] = bool(args.use_bn)

    cfg["training"]["trainer"] = args.trainer

    if args.lr_rate:
        cfg['training']['scheduler']['init_lr'] = args.lr_rate

    if args.trainer in ["cdan", "mcc", "memsac", "ilada", "bsp"]:
        cfg["model"]["discriminator"]["in_feature"] *= n_class ## for cdan
    elif args.trainer in ["hda", "toalign"]:
        cfg["model"]["discriminator"]["in_feature"] = n_class ## for hdan
    cfg["training"]["freeze_encoder"] = args.linear
        

    cfg["data"]["source"]["data_root"] = args.data_root
    cfg["data"]["target"]["data_root"] = args.data_root
    cfg["data"]["source"]["train"] = cfg["data"]["source"]["val"] = args.source
    cfg["data"]["target"]["train"] = cfg["data"]["target"]["val"] = args.target

    ## for domain net, seperate test set
    if args.target_test is not None:
        test_file_target = args.target_test
    else:
        test_file_target = os.path.join(os.getcwd(), args.target.replace("train.txt" , "test.txt"))

    if args.source_test is not None:
        test_file_source = args.source_test
    else:
        test_file_source = os.path.join(os.getcwd(), args.source.replace("train.txt" , "test.txt"))

    if os.path.exists(test_file_target):
        cfg["data"]["target"]["val"] = test_file_target
    if os.path.exists(test_file_source):
        cfg["data"]["source"]["val"] = test_file_source

    if args.cbs_source:
        cfg["data"]["source"]["sampler"] = {"name" : "class_balanced"}
    else:
        cfg["data"]["source"]["sampler"] = {"name" : "random"}

    cfg["data"]["source"]["imbalance_factor"] = abs(args.source_imb_factor)/100
    if args.source_imb_factor < 0:
        cfg["data"]["source"]["reversed"] = True
    cfg["data"]["target"]["imbalance_factor"] = abs(args.target_imb_factor)/100
    if args.target_imb_factor < 0:
        cfg["data"]["target"]["reversed"] = True

    cfg["data"]["source"]["mode"] = args.prune_mode
    cfg["data"]["target"]["mode"] = args.prune_mode

    cfg['training']['batch_size'] = args.batch_size
    cfg["model"]["feature_extractor"]["arch"] = args.backbone

    if args.backbone in ["timm_swin"]: ## different crop sizes only for swin
        cfg["data"]["source"]["crop_size"] = cfg["data"]["target"]["crop_size"] = 256
    else:
        cfg["data"]["source"]["crop_size"] = cfg["data"]["target"]["crop_size"] = 224


    if "resnet" in args.backbone:
        cfg["model"]["classifier"]["feat_size"] = [2048,256]
    elif args.backbone == "vits16":
        cfg["model"]["classifier"]["feat_size"] = [384,256]
    elif args.backbone.startswith(("vitb16")):
        cfg["model"]["classifier"]["feat_size"] = [768,256]
    elif args.backbone.startswith(("vitl16")):
        cfg["model"]["classifier"]["feat_size"] = [1024,256]
    elif args.backbone in ["timm_swin", "timm_convnext"]:
        cfg["model"]["classifier"]["feat_size"] = [768,256]
    elif args.backbone in ["timm_deit", "timm_resmlp"]:
        cfg["model"]["classifier"]["feat_size"] = [384,256]

    if args.trainer == "mdd":
        d1, d2 = cfg["model"]["classifier"]["feat_size"]
        cfg["model"]["classifier"]["feat_size"] = [d1,256,256]


    cfg["training"]["iteration"] = args.num_iter
    cfg["exp"] = args.exp_name

    ## Method specific parameters
    if cfg["training"]["trainer"] == "memsac":
        cfg["training"]["losses"]["loss_msc"]["dim"] = cfg["model"]["classifier"]["feat_size"][-1]

    logdir = os.path.join('runs', os.path.basename(args.config)[:-4], cfg['exp'])
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Start logging')

    logger.info(args)

    main()

