# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt

class TensorRGBToGray:
    def __init__(self, probability=1.0):
        self.probability = probability

    def __call__(self, img_tensor):
        if random.random() < self.probability:
            # 将 RGB 图像转换为灰度图像
            gray_tensor = img_tensor.mean(dim=0, keepdim=True)
            # 将灰度图像复制到三个通道
            return gray_tensor.repeat(3, 1, 1)
        return img_tensor


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


def read_two_images(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            #img1 = img
            img1 = img.crop((0, 0, 256, 128))
            img2 = img.crop((256, 0, 512, 128))
            #print(img1)
            #print(img2)
            #pdb.set_trace()
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img1, img2

def show_images(img1, img2):
    """Display two images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1)
    axes[0].set_title("Image 1")
    axes[0].axis('off')
    
    axes[1].imshow(img2)
    axes[1].set_title("Image 2")
    axes[1].axis('off')
    
    plt.show()

class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path

class RGBIRDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        #pdb.set_trace()
        img1, img2= read_two_images(img_path)
        #print(img1)
        #T.functional.crop(img,0,0,640,360)
        #print(img)
        #pdb.set_trace()
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        #print(img1)
        #pdb.set_trace()
        
        return (img1, img2), pid, camid, img_path

class MSVR310Dataset(Dataset):
    """For msvr310"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        #pdb.set_trace()
        img1 = read_image(img_path[0])
        img2 = read_image(img_path[1])
        #print(img1)
        #T.functional.crop(img,0,0,640,360)
        #print(img)
        #pdb.set_trace()
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        #print(img1)
        #pdb.set_trace()
        return (img1, img2), pid, camid, img_path

class WMVeID863Dataset(Dataset):
    """For wmveid863"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        #pdb.set_trace()
        img1 = read_image(img_path[0])
        img2 = read_image(img_path[1])
        #print(img1)
        #T.functional.crop(img,0,0,640,360)
        #print(img)
        #pdb.set_trace()
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        #print(img1)
        #pdb.set_trace()
        return (img1, img2), pid, camid, img_path

class CMShipDataset(Dataset):
    """For CMShip"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        #pdb.set_trace()
        img1 = read_image(img_path[0])
        img2 = read_image(img_path[1])
        #print(img1)
        #T.functional.crop(img,0,0,640,360)
        #print(img)
        #pdb.set_trace()
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        #print(img1)
        #pdb.set_trace()
        return (img1, img2), pid, camid, img_path


if __name__ == '__main__':
    img1, img2 = read_two_images(r'dataset\rgbir\bounding_box_train\0001_c0001_003.jpg')
    show_images(img1, img2)