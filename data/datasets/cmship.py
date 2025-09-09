# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re
import pdb
import os
import os.path as osp
import numpy as np
from .bases import BaseImageDataset

class CMShip(BaseImageDataset):
    """
    URL: https://www.scidb.cn/detail?dataSetId=d1f09cc8221d43438b37e931e1fbec25&version=V1

    """
    dataset_dir = 'CMshipReID-dataset'

    def __init__(self, root='/home/haoluo/data', verbose=True, **kwargs):
        super(CMShip, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_ids_dir = osp.join(self.dataset_dir, 'exp', 'train_id.txt')
        self.val_ids_dir = osp.join(self.dataset_dir, 'exp', 'val_id.txt')
        self.test_ids_dir = osp.join(self.dataset_dir, 'exp', 'test_id.txt')

        self.train_ids, self.test_ids = self._process_ids(self.train_ids_dir, self.val_ids_dir, self.test_ids_dir)

        train = self._process_dir(self.train_ids, relabel=True)
        query = self._process_dir(self.test_ids, relabel=False)
        gallery = self._process_dir(self.test_ids, relabel=False)
        #pdb.set_trace()
        if verbose:
            print("=> CMShip loaded")
            self.print_dataset_statistics(train, query, gallery)
        
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _process_ids(self, train_ids_dir, val_ids_dir, test_ids_dir):
        def _read_ids(file_path):
            with open(file_path, 'r') as f:
                ids = f.readline().strip()
            return [f'{int(id_.strip()):03d}' for id_ in ids.split(',') if id_.strip()]

        train_ids = _read_ids(train_ids_dir)
        val_ids = _read_ids(val_ids_dir)
        test_ids = _read_ids(test_ids_dir)
        return train_ids + val_ids, test_ids

    def _process_dir(self, id_list, relabel=False):
        vid_container = set()
        for vid in id_list:
            vid_container.add(int(vid))
        vid2label = {vid: label for label, vid in enumerate(vid_container)}

        dataset = []
        img_path = osp.join(self.dataset_dir, 'CMShipReID')
        for vid in id_list:
            r_data = os.listdir(osp.join(img_path, 'VIS', vid))
            for img in  r_data:
                r_img_path = osp.join(img_path, 'VIS', vid, img)
                n_img_path = osp.join(img_path, 'NIR', vid, img)
                t_img_path = osp.join(img_path, 'TIR', vid, img)
                id = int(vid)
                camid = 1
                if relabel:
                    id = vid2label[id]
                dataset.append(((r_img_path, n_img_path, t_img_path), id, camid))
        return dataset


if __name__ == '__main__':
    dataset = CMShip(root=r'E:\ReID\datasets')
    print(dataset.train_ids)
    print(dataset.test_ids)