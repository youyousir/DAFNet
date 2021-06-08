import torch
import torch.nn as nn
from DeformCN.modules.deform_conv import DeformConvPack, DeformableAlign, BiDeformableFusion, DeformableAlign1


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum, **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.ConvTranspose2d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def group_weight(group_decay, group_no_decay, module):
    l1 = len(group_decay)
    l2 = len(group_no_decay)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Conv1d, BiDeformableFusion, DeformConvPack, DeformableAlign, DeformableAlign1, nn.ConvTranspose2d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.PReLU):
            group_decay.append(m.weight)


        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, (
        nn.GroupNorm, nn.InstanceNorm2d, nn.LayerNorm, nn.SyncBatchNorm, nn.BatchNorm1d)):

            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    if len(list(module.parameters())) != len(group_decay) + len(group_no_decay) - l1 -l2:
        print(module)

    return group_decay, group_no_decay
