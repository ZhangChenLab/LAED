"""Pytorch dataset object that loads MNIST dataset as bags."""

from __future__ import print_function

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

import argparse

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable

# from resnet50_raw import resnet50
import cv2
import os
import glob
from torchvision import transforms, utils
from PIL import Image
import GlobalManager as gm
import albumentations as A
import torchvision.utils as vutils

path=gm.get_value('path')


normalize = transforms.Normalize(
    mean=[0.15, 0.35, 0.35],
    std=[0.25, 0.26, 0.28]
)    

train_preprocess = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    # normalize,
]) 

test_preprocess = transforms.Compose([
    # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
    transforms.ToTensor(),
    normalize
]) 

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, brightness_by_max=True, always_apply=False, p=0.5),
    A.GaussNoise (var_limit=(50.0, 300.0), mean=0, always_apply=False, p=0.2),
    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.05, 0.3), always_apply=False, p=0.2),
    A.MultiplicativeNoise (multiplier=(0.9, 1.1), per_channel=False, elementwise=False, always_apply=False, p=0.2),
    A.Downscale (scale_min=0.75, scale_max=0.9, interpolation=0, always_apply=False, p=0.2)
    ],)
       
        
def MarrowLoader(path):
    img_pil =  Image.open(path).convert('RGB')
    img_pil = img_pil.resize((224,224))
    # fn=str(np.random.randint(1000))+'.png'
    # img_pil.save('H:/data/bone_marrow/k_bag_618/patch/'+fn, quality=95)
    # print(img_pil.size)
    img_tensor = train_preprocess(img_pil)
    # img_np=img_tensor.numpy()
    # img_np=img_np-img_np.min()
    # img_np=np.uint8(img_np/img_np.max()*255)
    # print(img_np.shape)
    # img_np = Image.fromarray(img_np)
    # img_t = img_tensor.unsqueeze(0)

    # 将张量保存为图片文件
    # vutils.save_image(img_t, 'H:/data/bone_marrow/k_bag_618/patch/'+fn, normalize=True)
    # img_tensor.save('H:/data/bone_marrow/k_bag_618/patch/'+fn)
   
    return img_tensor
def TestLoader(path):
    img_pil =  Image.open(path).convert('RGB')
    img_pil = img_pil.resize((224,224))
    img_tensor = test_preprocess(img_pil)
    return img_tensor
class trainset(Dataset):
    def __init__(self, loader=MarrowLoader):
        self.loader = loader
        self.rd=np.random.RandomState()
    def __getitem__(self, index):
        train_folder=gm.get_value('train_folder')
        target=0
        im_list=glob.glob(os.path.join(train_folder,'*.tif'))
        index=np.random.randint(0,len(im_list),1)[0]
        fn = im_list[index]
        img = self.loader(fn)
        return img,target
    def __len__(self):
        return 64*2000
    
class testset(Dataset):
    def __init__(self, loader=TestLoader):
        self.loader = loader
        self.rd=np.random.RandomState()
        self.test_image=[]
    def __getitem__(self, index):
        train_folder=gm.get_value('train_folder')
        target=0
        im_list=glob.glob(os.path.join(train_folder,'*.tif'))
        index=np.random.randint(0,len(im_list),1)[0]
        fn = im_list[index]
        self.test_image.append(fn)
        img = self.loader(fn)
        return img,target
    def __len__(self):
        gm.set_value("test_image",self.test_image)
        return 1000
    
