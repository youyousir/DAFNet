#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
from logger import setup_logger
from CGAN.model import DAFNet
from cityscapes import CityScapes
from CGAN.loss import OhemCELoss
# from CGAN.init_func import init_weight
from optimizer import Optimizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import os.path as osp
import logging
import time
import datetime
import argparse
from utils.score import SegmentationMetric
from tabulate import tabulate


respth = '/output/result'
if not osp.exists(respth):
    os.makedirs(respth, exist_ok=True)
logger = logging.getLogger()


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--local_rank',
            dest = 'local_rank',
            type = int,
            default = -1,
            )
    return parse.parse_args()




def train():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
                backend='nccl',
        init_method='env://'
#                 init_method='tcp://127.0.0.1:16222',
#                 world_size=torch.cuda.device_count(),
#                 rank=args.local_rank
                )
    setup_logger(respth)

    ## dataset
    n_classes = 19
    n_img_per_gpu = 8
    n_workers = 4
    cropsize = [1024, 1024]
    ds = CityScapes('/data/bitahub/Cityscapes', cropsize=cropsize, mode='train')
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
                    batch_size=n_img_per_gpu,
                    shuffle=False,
                    sampler=sampler,
                    num_workers=n_workers,
                    pin_memory=True,
                    drop_last=True)
    

    ## model
    ignore_idx = 255
    net = DAFNet(n_classes=n_classes)
    
#     net_state_dict = net.state_dict() 
#     save_pth = osp.join('/model/yoyosir/resnet18-cityscape', 'DAFNet3.pth')
#     pretrained_dict = torch.load(save_pth)
#     pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
#     net_state_dict.update(pretrained_dict_1)
#     net.load_state_dict(net_state_dict)
#     print('******pretrained model loaded!******')


    net.cuda()
    net.train()
    net = nn.parallel.DistributedDataParallel(net,
            device_ids = [args.local_rank, ],
            output_device = args.local_rank,
#             find_unused_parameters=True
            )
    score_thres = 0.7
    n_min = n_img_per_gpu * cropsize[0] * cropsize[1] // 16
    Loss1 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
    Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
#     Loss3 = OhemCrossEntropy2dTensor(use_weight=True)


    ## optimizer
    momentum = 0.9
    weight_decay = 5e-4
    lr_start = 1e-2
    max_iter = 40000 
    power = 0.9
    warmup_steps = 1000
    warmup_start_lr = 1e-5
    optim = Optimizer(
            model = net.module,
            lr0 = lr_start,
            momentum = momentum,
            wd = weight_decay,
            warmup_steps = warmup_steps,
            warmup_start_lr = warmup_start_lr,
            max_iter = max_iter,
            power = power)
    
    ## train loop
    msg_iter = 50
    loss1_avg = []
    loss2_avg = []
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    epoch = 0
    for it in range(max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0]==n_img_per_gpu: raise StopIteration
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()
        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        fine2, fine, coarse = net(im)
        loss1 = Loss1(coarse, lb)
        loss2 = Loss2(fine, lb)
        loss3 = Loss3(fine2, lb)
         
        loss = 0.4*loss1 + 0.6*loss2 + loss3

        loss.backward()
        optim.step()

        loss_avg.append(loss1.item())
        loss1_avg.append(loss2.item())
        loss2_avg.append(loss3.item())
        
        ## print training log message
        if (it+1)%msg_iter==0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            loss1_avg = sum(loss1_avg) / len(loss1_avg)
            loss2_avg = sum(loss2_avg) / len(loss2_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds=eta))
            msg = ', '.join([
                    'it: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'loss1: {loss1:.4f}',
                    'loss2: {loss2:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]).format(
                    it = it+1,
                    max_it = max_iter,
                    lr = lr,
                    loss = loss_avg,
                    loss1 = loss1_avg,
                    loss2 = loss2_avg,
                    time = t_intv,
                    eta = eta
                )
            logger.info(msg)
            loss_avg = []
            loss1_avg = []
            loss2_avg = []
            st = ed
            
            
        if (it+1)%10000==0:
            save_pth = osp.join(respth, 'model_iter_'+ str(it)+'.pth')
            state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            if dist.get_rank()==0: torch.save(state, save_pth)

    ## dump the final model
    save_pth = osp.join(respth, 'model_CG.pth')
    net.cpu()
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    if dist.get_rank()==0: torch.save(state, save_pth)
    logger.info('training done, model saved to: {}'.format(save_pth))
    
    


if __name__ == "__main__":
    train()
#     evaluate()
