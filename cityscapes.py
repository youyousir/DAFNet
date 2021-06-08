
#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json
from cvtransform import *



class CityScapes(Dataset):
    def __init__(self, data_path, list_path='/code/DAFNet/dataset/', cropsize=(1024, 1024), mode='train', *args, **kwargs):
        super(CityScapes, self).__init__()
        assert mode in ('train', 'trainval', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255
        self.data_path = data_path
        self.list = 'cityscapes_' + mode + '_list.txt'
        self.list_path = osp.join(list_path, self.list)
        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.data_path, name.split()[0])
            label_file = osp.join(self.data_path, name.split()[1])
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name.split('/')[-1]
            })

        print('length of {} set is: {:.0f}' .format(self.mode, len(self.files)))
        self.len = len(self.files)

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.trans_train = Compose([
            ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.25),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize)
        ])

    def __getitem__(self, index):
        datafiles = self.files[index]
        fn = datafiles['name']
        img = Image.open(datafiles['img'])#.convert('RGB')
        label = Image.open(datafiles['label'])
        if self.mode == 'trainval' or self.mode == 'train':
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return img, label

    def __len__(self):
        return self.len


class CityScapes2(Dataset):
    def __init__(self, data_path, list_path='/code/Seg/daset/', mode='test', *args, **kwargs):
        super(CityScapes2, self).__init__()
        assert mode in ('test', 'val')
        self.mode = mode
        self.ignore_lb = 255
        self.data_path = data_path
        self.list = 'cityscapes_' + mode + '_list.txt'
        self.list_path = osp.join(list_path, self.list)
        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        self.files = []

        for name in self.img_ids:
            img_file = osp.join(self.data_path, name.split()[0])
#             img_file = osp.join(self.data_path, name)
            self.files.append({
                "img": img_file,
                "name": name.split('/')[-1].split('.')[0]
            })

        print('length of {} set is: {:.0f}'.format(self.mode, len(self.files)))
        self.len = len(self.files)

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, index):
        datafiles = self.files[index]
        fn = datafiles['name']
        img = Image.open(datafiles['img'])# .convert('RGB')
        img = self.to_tensor(img)
        return img, fn

    def __len__(self):
        return self.len

    
    
    
class CityScapestest(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='test', *args, **kwargs):
        super(CityScapestest, self).__init__()
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255

        # with open('./cityscapes_info.json', 'r') as fr:
        #     labels_info = json.load(fr)
        # self.lb_map = {el['id']: el['trainId'] for el in labels_info}

        ## parse img directory
        self.imgs = {}
        imgnames = []
        impth = osp.join(rootpth, 'leftImg8bit', mode)
        folders = os.listdir(impth)
        for fd in folders:
            fdpth = osp.join(impth, fd)
            im_names = os.listdir(fdpth)
            names = [el.replace('_leftImg8bit.png', '') for el in im_names]
            impths = [osp.join(fdpth, el) for el in im_names]
            imgnames.extend(names)
            self.imgs.update(dict(zip(names, impths)))

        ## parse gt directory
        # self.labels = {}
        # gtnames = []
        # gtpth = osp.join(rootpth, 'gtFine', mode)
        # folders = os.listdir(gtpth)
        # for fd in folders:
        #     fdpth = osp.join(gtpth, fd)
        #     lbnames = os.listdir(fdpth)
        #     lbnames = [el for el in lbnames if 'labelIds' in el]
        #     names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
        #     lbpths = [osp.join(fdpth, el) for el in lbnames]
        #     gtnames.extend(names)
        #     self.labels.update(dict(zip(names, lbpths)))

        self.imnames = imgnames
        self.len = len(self.imnames)
        # assert set(imgnames) == set(gtnames)
        assert set(self.imnames) == set(self.imgs.keys())
        # assert set(self.imnames) == set(self.labels.keys())

        ## pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)), 
            RandomCrop(cropsize)
            ])


    def __getitem__(self, idx):
        fn = self.imnames[idx]
        impth = self.imgs[fn]
        # lbpth = self.labels[fn]
        img = Image.open(impth)
        # label = Image.open(lbpth)
        # if self.mode == 'train':
        #     im_lb = dict(im=img, lb=label)
        #     im_lb = self.trans_train(im_lb)
        #     img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        # label = np.array(label).astype(np.int64)[np.newaxis, :]
        # label = self.convert_labels(label)
        return img, fn


    def __len__(self):
        return self.len

# class CityScapessingle(Dataset):
#     def __init__(self, rootpth, cropsize=(640, 480), mode='test', *args, **kwargs):
#         super(CityScapessingle, self).__init__()
#         assert mode in ('train', 'val', 'test')
#         self.mode = mode
#         self.ignore_lb = 255

#         # with open('./cityscapes_info.json', 'r') as fr:
#         #     labels_info = json.load(fr)
#         # self.lb_map = {el['id']: el['trainId'] for el in labels_info}

#         ## parse img directory
#         self.imgs = {}
#         imgnames = []
#         impth = osp.join(rootpth, 'leftImg8bit', mode)
#         folders = os.listdir(impth)
#         for fd in folders:
#             fdpth = osp.join(impth, fd)
#             im_names = os.listdir(fdpth)
#             names = [el.replace('_leftImg8bit.png', '') for el in im_names]
#             impths = [osp.join(fdpth, el) for el in im_names]
#             imgnames.extend(names)
#             self.imgs.update(dict(zip(names, impths)))

#         ## parse gt directory
#         # self.labels = {}
#         # gtnames = []
#         # gtpth = osp.join(rootpth, 'gtFine', mode)
#         # folders = os.listdir(gtpth)
#         # for fd in folders:
#         #     fdpth = osp.join(gtpth, fd)
#         #     lbnames = os.listdir(fdpth)
#         #     lbnames = [el for el in lbnames if 'labelIds' in el]
#         #     names = [el.replace('_gtFine_labelIds.png', '') for el in lbnames]
#         #     lbpths = [osp.join(fdpth, el) for el in lbnames]
#         #     gtnames.extend(names)
#         #     self.labels.update(dict(zip(names, lbpths)))

#         self.imnames = imgnames
#         self.len = len(self.imnames)
#         # assert set(imgnames) == set(gtnames)
#         assert set(self.imnames) == set(self.imgs.keys())
#         # assert set(self.imnames) == set(self.labels.keys())

#         ## pre-processing
#         self.to_tensor = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#             ])
#         self.trans_train = Compose([
#             ColorJitter(
#                 brightness=0.5,
#                 contrast=0.5,
#                 saturation=0.5),
#             HorizontalFlip(),
#             RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
#             RandomCrop(cropsize)
#             ])


#     def __getitem__(self, idx):
#         fn = self.imnames[idx]
#         impth = self.imgs[fn]
#         # lbpth = self.labels[fn]
#         img = Image.open(impth)
#         # label = Image.open(lbpth)
#         # if self.mode == 'train':
#         #     im_lb = dict(im=img, lb=label)
#         #     im_lb = self.trans_train(im_lb)
#         #     img, label = im_lb['im'], im_lb['lb']
#         img = self.to_tensor(img)
#         # label = np.array(label).astype(np.int64)[np.newaxis, :]
#         # label = self.convert_labels(label)
#         return img, fn


#     def __len__(self):
#         return self.len

    
    

if __name__ == "__main__":
    from tqdm import tqdm
    ds = CityScapes(rootpth='/Volumes/myMac2/dataset/cityscapes/', mode='val', n_classes=19)
    uni = []
    for im, lb in tqdm(ds):
        lb_uni = np.unique(lb).tolist()
        uni.extend(lb_uni)
    print(uni)
    print(set(uni))

