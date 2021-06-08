#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class CrossEntropyLoss2d(nn.Module):
    '''
    This file defines a cross entropy loss for 2D images
    '''

    def __init__(self, ignore_label=255):
        '''
        :param weight: 1D weight vector to deal with the class-imbalance
        Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network. 
        You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.
        '''
        super().__init__()

        # self.loss = nn.NLLLoss2d(weight, ignore_index=255)
        weight = torch.cuda.FloatTensor([4.4286523, 3.443227 , 9.627174 , 2.9082294, 6.9883366, 5.2119217,
       9.490196 , 9.150478 , 7.1654587, 9.888601 , 7.5455384])
        self.loss = nn.NLLLoss(weight, ignore_index=ignore_label)

    def forward(self, outputs, targets):
        return self.loss(F.log_softmax(outputs, 1), targets)


class IOU_re(nn.Module):
    def __init__(self, ignore_index=255, num_classes=19):
        super(IOU_re, self).__init__()
        self.ignore_idex = ignore_index
        self.num_classes = num_classes

    def forward(self, logits, labels):
        B = logits.size()[0]
        labels = F.interpolate(labels.unsqueeze(1).to(torch.float32), size=logits.size()[2:], mode='nearest')
        labels = labels.squeeze(1).to(torch.int64)
        labels[labels==self.ignore_idex] = self.num_classes      #  label   B, H, W

        preds = torch.argmax(logits, dim=1)   # pred  B, H, W
        errors = preds - labels      #
        errors[errors != 0] = 1  # 正确的是0， 错误的是1
        errors = errors.unsqueeze(1)  # 我们希望各个类，错误比例最小化。  B, 1, H,W

        labels = F.one_hot(labels, self.num_classes+1)[:, :, :, 0:self.num_classes].permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        errors = errors * labels     # B，C，H，W

        i = torch.sum(errors.view(B, self.num_classes, -1), dim=2)
        u = torch.sum(labels.view(B, self.num_classes, -1), dim=2)

        i = i.to(torch.float32)
        u = u.to(torch.float32)
        
        num = len(u[u == 0])
        u[u==0] = 1
        re = torch.sum(i/u) / (self.num_classes * B - num)

        return re

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)
    
class OhemCrossEntropy2dTensor(nn.Module):
    """
        Ohem Cross Entropy Tensor Version
    """
    def __init__(self, ignore_index=255, thresh=0.7, min_kept=10000,
                 use_weight=False):
        super(OhemCrossEntropy2dTensor, self).__init__()
        self.ignore_index = ignore_index
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            weight = torch.cuda.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction="mean",
                                                       weight=weight,
                                                       ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction="mean",
                                                       ignore_index=ignore_index)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_index)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            print('Labels: {}'.format(num_valid))
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                _, index = mask_prob.sort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask

        target = target.masked_fill_(~valid_mask, self.ignore_index)
        target = target.view(b, h, w)

        return self.criterion(pred, target)
    
    
class OhemCELossW(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELossW, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        weight = torch.cuda.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
        self.criteria = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self.loss = nn.SmoothL1Loss()

    def forward(self, pred, cluster):
        if pred.size() != cluster.size():
            cluster = F.interpolate(cluster, size=pred.size()[2:], mode='nearest')
        N, C, H, W = pred.size()
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, C)
        cluster = cluster.permute(0, 2, 3, 1).contiguous().view(-1, C)
        mask = torch.sum(cluster, dim=1, keepdim=True)
        mask[mask!=0] = 1
        pred = pred * mask
        loss = self.loss(pred, cluster)
        return loss

    
    
class ProbOhemCrossEntropy2d(nn.Module):
    def __init__(self, ignore_label, reduction='mean', thresh=0.6, min_kept=256,
                 down_ratio=1, use_weight=True):
        super(ProbOhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        self.down_ratio = down_ratio
        if use_weight:
            weight = torch.cuda.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955,
                 1.0865, 1.1529, 1.0507])
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       weight=weight,
                                                       ignore_index=ignore_label)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction=reduction,
                                                       ignore_index=ignore_label)

    def forward(self, pred, target):
        b, c, h, w = pred.size()
        target = target.view(-1)
        valid_mask = target.ne(self.ignore_label)
        target = target * valid_mask.long()
        num_valid = valid_mask.sum()

        prob = F.softmax(pred, dim=1)
        prob = (prob.transpose(0, 1)).reshape(c, -1)

        if self.min_kept > num_valid:
            # logger.info('Labels: {}'.format(num_valid))
            pass
        elif num_valid > 0:
            prob = prob.masked_fill_(~valid_mask, 1)
            mask_prob = prob[
                target, torch.arange(len(target), dtype=torch.long)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = mask_prob.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if mask_prob[threshold_index] > self.thresh:
                    threshold = mask_prob[threshold_index]
                kept_mask = mask_prob.le(threshold)
                target = target * kept_mask.long()
                valid_mask = valid_mask * kept_mask
                # logger.info('Valid Mask: {}'.format(valid_mask.sum()))

        target = target.masked_fill_(~valid_mask, self.ignore_label)
        target = target.view(b, h, w)

        return self.criterion(pred, target)
    
    
    
    
class KLloss(nn.Module):
    def __init__(self):
        super(KLloss, self).__init__()

    def forward(self, feat, cfeat):
        if feat.size() != cfeat.size():
            feat = F.interpolate(feat, size=cfeat.size()[2:], mode='nearest')
        N, C, H, W = feat.size()
        cfeat = cfeat.permute(0, 2, 3, 1).contiguous().view(-1, C)
        feat = feat.permute(0, 2, 3, 1).contiguous().view(-1, C)
        mask = torch.sum(cfeat, dim=1, keepdim=True)
        mask[mask!=0] = 1
        soft_T = F.softmax(cfeat, dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        loss2 = (torch.sum((soft_T * logsoftmax(cfeat) - soft_T * logsoftmax(feat))*mask)) / W / H / N
        return loss2
    
    
if __name__ == '__main__':
    torch.manual_seed(15)
    criteria1 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    criteria2 = OhemCELoss(thresh=0.7, n_min=16*20*20//16).cuda()
    net1 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net1.cuda()
    net1.train()
    net2 = nn.Sequential(
        nn.Conv2d(3, 19, kernel_size=3, stride=2, padding=1),
    )
    net2.cuda()
    net2.train()

    with torch.no_grad():
        inten = torch.randn(16, 3, 20, 20).cuda()
        lbs = torch.randint(0, 19, [16, 20, 20]).cuda()
        lbs[1, :, :] = 255

    logits1 = net1(inten)
    logits1 = F.interpolate(logits1, inten.size()[2:], mode='bilinear')
    logits2 = net2(inten)
    logits2 = F.interpolate(logits2, inten.size()[2:], mode='bilinear')

    loss1 = criteria1(logits1, lbs)
    loss2 = criteria2(logits2, lbs)
    loss = loss1 + loss2
    print(loss.detach().cpu())
    loss.backward()
