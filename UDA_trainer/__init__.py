from .cdan import train_cdan
from .plain import train_plain
from .dann import train_dann
from .safn import train_safn
from .mcc import train_mcc
from .memsac import train_memsac
from .mdd import train_mdd
from .mcd import train_mcd
from .toalign import train_toalign
from .adamatch import train_adamatch
from .daln import train_daln
from .bsp import train_bsp
from .val import val
from .eval import eval

def get_trainer(cfg):
    trainer = _get_trainer_instance(cfg['trainer'])
    return trainer


def _get_trainer_instance(name):
    try:
        return {
            'plain' : train_plain,
            'cdan' : train_cdan,
            'dann' : train_dann,
            'safn' : train_safn,
            'mcc' : train_mcc,
            'memsac' : train_memsac,
            'ilada' : train_memsac,
            'mdd' : train_mdd,
            'mcd' : train_mcd,
            "toalign" : train_toalign,
            "adamatch" : train_adamatch,
            "daln"     : train_daln,
            "bsp"      : train_bsp,
        }[name]
    except:
        raise BaseException('Trainer type {} not available'.format(name))


