# encoding: utf-8

from .rgbir import Rgbir
from .msvr310 import MSVR310
from .wmveid863 import WMVeID863
from .cmship import CMShip
from .dataset_loader import ImageDataset, RGBIRDataset, MSVR310Dataset, WMVeID863Dataset, CMShipDataset

__factories = {
    'rgbir': Rgbir,
    'msvr310': MSVR310,
    'wmveid863': WMVeID863,
    'cmship': CMShip,
}

def get_names():
    return __factories.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factories.keys():
        raise KeyError("Unknown dataset: {}", name)
    return __factories[name](*args, **kwargs)