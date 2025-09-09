# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, visible_collate_fn, infrared_collate_fn, visual_collate_fn
from .datasets import init_dataset, ImageDataset, RGBIRDataset, MSVR310Dataset, WMVeID863Dataset, CMShipDataset
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms
import pdb

dataset_map = {
    "rgbir": RGBIRDataset,
    "msvr310": MSVR310Dataset,
    "wmveid863": WMVeID863Dataset,
    "cmship": CMShipDataset,
}

def make_data_loader(cfg, reconst=False):
    if reconst:
        train_transforms = build_transforms(cfg, is_train=False)
    else:
        train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids

    dataset_name = cfg.DATASETS.NAMES.lower()
    DatasetClass = dataset_map.get(dataset_name)

    if DatasetClass is None:
        raise ValueError(f"Unsupported dataset name: {cfg.DATASETS.NAMES}")

    train_set = DatasetClass(dataset.train, train_transforms)
    query_set = DatasetClass(dataset.query, val_transforms)
    gallery_set = DatasetClass(dataset.gallery, val_transforms)
    
    #pdb.set_trace()
    if reconst:
        if cfg.DATALOADER.SAMPLER == 'softmax':
            train_loader = DataLoader(
                train_set, batch_size=cfg.DATALOADER.VISUAL_NUM_INSTANCE, shuffle=True, num_workers=num_workers,
                collate_fn=visual_collate_fn, drop_last=True
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMGS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMGS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
                num_workers=num_workers, collate_fn=visual_collate_fn, drop_last=True
            )
        # pdb.set_trace()
        query_loader = DataLoader(
            query_set, batch_size=cfg.TEST.IMGS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=visual_collate_fn, drop_last=False
        )
        gallery_loader = DataLoader(
            gallery_set, batch_size=cfg.TEST.IMGS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=visual_collate_fn, drop_last=False
        )
    else:
        if cfg.DATALOADER.SAMPLER == 'softmax':
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMGS_PER_BATCH, shuffle=True, num_workers=num_workers,
                collate_fn=train_collate_fn, drop_last=True
            )
        else:
            train_loader = DataLoader(
                train_set, batch_size=cfg.SOLVER.IMGS_PER_BATCH,
                sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMGS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
                num_workers=num_workers, collate_fn=train_collate_fn, drop_last=True
            )
    
        # pdb.set_trace()
        if cfg.SOLVER.MODE == [1, 2]:   # RGB2IR
            query_loader = DataLoader(
                query_set, batch_size=cfg.TEST.IMGS_PER_BATCH, shuffle=False, num_workers=num_workers,
                collate_fn=visible_collate_fn, drop_last=False
            )
            gallery_loader = DataLoader(
                gallery_set, batch_size=cfg.TEST.IMGS_PER_BATCH, shuffle=False, num_workers=num_workers,
                collate_fn=infrared_collate_fn, drop_last=False
            )
        else:   # IR2RGB
            query_loader = DataLoader(
                query_set, batch_size=cfg.TEST.IMGS_PER_BATCH, shuffle=False, num_workers=num_workers,
                collate_fn=infrared_collate_fn, drop_last=False
            )
            gallery_loader = DataLoader(
                gallery_set, batch_size=cfg.TEST.IMGS_PER_BATCH, shuffle=False, num_workers=num_workers,
                collate_fn=visible_collate_fn, drop_last=False
            )
    #pdb.set_trace()
    return dataset, train_loader, query_loader, gallery_loader, len(dataset.query), len(dataset.gallery), num_classes
