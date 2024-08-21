import copy
import torch.nn as nn
import logging

from .models_timm import timm_convnext, timm_deit, timm_resmlp, timm_resnet50, timm_swin

from .resnet import resnet10, resnet101, resnet18, resnet50
from .vit_timm import vit_b_16, vit_s_16, vit_l_16
from .vit_swag import ViTB16_swag, ViTL16_swag
from .linearcls import linearcls
from .lenet import lenet
from .mlpcls import mlpcls
from .fscls import fscls
from .advnet import advnet
from .mlpmdd import mddcls
from .mlpmcd import mcdcls
from .mlphda import hdacls
from .randomlyr import randomlayer

from .utils import grl_hook
from .memory_bank import MemoryModule


logger = logging.getLogger('mylogger')


def get_model(model_dict, verbose=False):

    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')

    if "timm" in name:
        if "resnet" in name:
            model = model(**param_dict)
            model.fc = nn.Identity() 
        elif "deit" in name or "resmlp" in name:
            model = model(**param_dict)
            model.head = nn.Identity() 
        elif "next" in name or "swin" in name:
            model = model(**param_dict)
            model.head.fc = nn.Identity() 
    else:
        if 'resnet' in name:
            model = model(**param_dict)
            model.fc = nn.Identity()
        elif 'vit' in name:
            model = model(**param_dict)
            model.head = nn.Identity()
        else:
            model = model(**param_dict)

    if verbose:
        logger.info(model)

    return model

def _get_model_instance(name):
    try:
        return {
            'resnet10': resnet10,
            'resnet18': resnet18,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'vitb16'   : vit_b_16,
            'vits16'   : vit_s_16,
            "vitl16"   : vit_l_16,
            "vitb16_swag" : ViTB16_swag,
            "vitl16_swag" : ViTL16_swag,
            'linearcls': linearcls,
            'mlpcls': mlpcls,
            'advnet': advnet,
            'randomlyr': randomlayer,
            'lenet': lenet,
            'fscls': fscls,
            'mddcls': mddcls,
            'mcdcls': mcdcls,
            'hdacls'   : hdacls,
            "timm_resnet50" : timm_resnet50,
            "timm_swin"     : timm_swin,
            "timm_convnext" : timm_convnext,
            "timm_resmlp"   : timm_resmlp,
            "timm_deit"     : timm_deit
        }[name]
    except:
        raise BaseException('Model {} not available'.format(name))


