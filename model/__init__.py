# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .baseline import embed_net
from .porsche import PORSCHE


def build_model(cfg, num_classes):
    if cfg.SOLVER.METHOD == 'base':
        print("Use Baseline Method!")
        model = embed_net(num_classes, no_local='off', gm_pool='off', arch=cfg.MODEL.NAME)
    elif cfg.SOLVER.METHOD == 'agw':
        print("Use AGW Method!")
        model = embed_net(num_classes, no_local='on', gm_pool ='on', arch=cfg.MODEL.NAME)
    elif cfg.SOLVER.METHOD == 'porsche':
        print("Use PORSCHE Method")
        model = PORSCHE(class_num=num_classes, arch=cfg.MODEL.NAME, final_layer=cfg.MODEL.FINAL_LAYER, 
                    neck=cfg.MODEL.NECK_DIM, nheads=cfg.MODEL.NUM_HEADS, num_encoder_layers=cfg.MODEL.NUM_TRANS_LAYERS - 1,
                    dim_feedforward=cfg.MODEL.D_FF, dropout=cfg.MODEL.DROPOUT_RATIO, pretrained=cfg.MODEL.USE_PRETRAIN)
    else:
        raise Exception("Method for {} is not supported".format(cfg.SOLVER.METHOD))
    return model