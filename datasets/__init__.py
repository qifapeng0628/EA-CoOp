from .btmri import BTMRI
from .busi import BUSI
from .ctkidney import CTKidney
from .kneexray import KneeXray
from .kvasir import Kvasir
from .lungcolon import LungColon
from .retina import RETINA
from .covid import COVID_19
from .dermamnist import DermaMNIST
from .octmnist import OCTMNIST
from .chmnist import CHMNIST
from .wbc import WBC
from .bach import BACH

dataset_list = {
                "BUSI": BUSI,
                "BTMRI": BTMRI,
                "CTKidney": CTKidney,
                "KneeXray": KneeXray,
                "Kvasir": Kvasir,
                "LungColon": LungColon,
                "RETINA": RETINA,
                "COVID_19": COVID_19,
                "DermaMNIST": DermaMNIST,
                "OCTMNIST": OCTMNIST,
                "CHMNIST": CHMNIST,
                "WBC": WBC,
                "BACH": BACH
                }


def build_dataset(cfg):
    return dataset_list[cfg.DATASET.NAME](cfg)