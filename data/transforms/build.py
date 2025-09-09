# encoding: utf-8
"""
@author:  liaoxingyu
@contact: liaoxingyu2@jd.com
"""

import torchvision.transforms as T

from .transforms import RandomErasing, ChannelAdapGray, ChannelRandomErasing

def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        transform_list = [
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,
            RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN),
        ]
        if cfg.INPUT.USE_CHANNEL_ERASE == True:
            transform_list.extend([
                ChannelRandomErasing(probability=cfg.INPUT.CHANNEL_RE_PROB, mean=cfg.INPUT.PIXEL_MEAN),
                ChannelAdapGray(probability=cfg.INPUT.ADAPT_GRAY_PROB)
            ])
        transform = T.Compose(transform_list)
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform
