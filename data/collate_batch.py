# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def getTensorData(imgs):
    batchsize = len(imgs)
    modalities = len(imgs[0])
    data = []
    for m in range(modalities):
        tmp = []
        for i in range(batchsize):
            tmp.append(imgs[i][m])
        data.append(torch.stack(tmp, dim=0))
    
    return data

def train_collate_fn(batch):
    imgs, pids, camids, img_path = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return getTensorData(imgs), pids, camids, img_path

def visible_collate_fn(batch):
    imgs, pids, camids, img_path = zip(*batch)
    return getTensorData(imgs)[0], pids, camids, img_path  

def infrared_collate_fn(batch):
    imgs, pids, camids, img_path = zip(*batch)
    return getTensorData(imgs)[1], pids, camids, img_path

def visual_collate_fn(batch):
    imgs, pids, camids, img_path = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return getTensorData(imgs), pids, camids, img_path