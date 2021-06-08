#!/usr/bin/python
# -*- encoding: utf-8 -*-
from logger import setup_logger
from CGAN.model import DAFNet
from cityscapes import CityScapes

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import numpy as np
from tqdm import tqdm
import math
from torch.autograd import Variable
from utils.score import SegmentationMetric
from tabulate import tabulate


def evaluate(respth='/model/yoyosir/resnet18-cityscape/', dspth='/data/bitahub/Cityscapes'):
    ## logger
    logger = logging.getLogger()

    ## model
    logger.info('\n')
    logger.info('===='*20)
    logger.info('evaluating the model ...\n')
    logger.info('setup and restore model')
    n_classes = 19
    net = DAFNet(n_classes=n_classes)
    
#     net_state_dict = net.state_dict() 
#     save_pth = osp.join('/model/yoyosir/resnet18-cityscape', 'CGANNet19.pth')
#     pretrained_dict = torch.load(save_pth)
#     pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
#     net_state_dict.update(pretrained_dict_1)
#     net.load_state_dict(net_state_dict)
    
    save_pth = osp.join(respth, 'DAFNet19.pth')
    net.load_state_dict(torch.load(save_pth))
    net.cuda()
    net.eval()

    ## dataset
    batchsize = 1
    n_workers = 4
    dsval = CityScapes(dspth, mode='val')
    dl = DataLoader(dsval,
                    batch_size = batchsize,
                    shuffle = False,
                    num_workers = n_workers,
                    drop_last = False)

    ## evaluator
    logger.info('compute the mIOU')
    # evaluator = MscEval(net, dl)

    ## eval
    meanIoU, per_class_iu = val(dl, net)
    print(meanIoU)
    print(per_class_iu)
    

    return 0




def val(val_loader, model):
    """
    args:
      val_loader: loaded for validation dataset
      model: model
    return: mean IoU and IoU class
    """
    model.eval()
    total_batches = len(val_loader)

    data_list = []
    metric = SegmentationMetric(nclass=19, distributed=False)

    for i, (image, target) in enumerate(val_loader):
        image = image.cuda()
        target = target.cuda()
        target[target==255] = -1
        with torch.no_grad():
            output, _, _= model(image)
        metric.update(output, target)
        pixAcc, mIoU = metric.get()

        print("Sample: {:d}, validation pixAcc: {:.3f}, mIoU: {:.3f}".format(
            i + 1, pixAcc * 100, mIoU * 100))

    pixAcc, mIoU, category_iou = metric.get(return_category_iou=True)

    print('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(
        pixAcc * 100, mIoU * 100))

    headers = ['class id', 'class name', 'iou']
    table = []
    classes = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
                'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
                'truck', 'bus', 'train', 'motorcycle', 'bicycle')
    for i, cls_name in enumerate(classes):
        table.append([cls_name, category_iou[i]])
    logging.info('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
                                                       numalign='center', stralign='center')))
    print('Category iou: \n {}'.format(tabulate(table, headers, tablefmt='grid', showindex="always",
                                                       numalign='center', stralign='center')))
    
    return mIoU, category_iou





if __name__ == "__main__":
#     setup_logger('./res')
    evaluate()
