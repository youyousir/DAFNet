'''
Modified from https://github.com/PkuRainBow/OCNet.pytorch/blob/master/oc_module/base_oc_block.py
'''

import torch
from torch import nn
from torch.nn import functional as F
from modules.bn import InPlaceABNSync as BatchNorm2d
# from torch.nn import BatchNorm2d

class ConvBN(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, groups=1, *args, **kwargs):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_chan, activation=None)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, groups=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_chan)
#         self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
#         x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
                    

class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


class CGANB(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels, value_channels, num_classes=19, psp_size=(1,3,6,8)):
        super(CGANB, self).__init__()
        self.in_channels = in_channels
        self.value_channels = value_channels
        self.out_channels = in_channels
        self.num_classes = num_classes
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        
        self.refine = nn.Sequential(
            ConvBNReLU(num_classes, num_classes, ks=3, padding=1),
            ConvBN(num_classes, num_classes, ks=1, padding=0)
        )


        self.psp = PSPModule(psp_size)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, feat, pred):
        batch_size, h, w = feat.size(0), feat.size(2), feat.size(3)
        if feat.size()[2:] != pred.size()[2:]:
            pred = F.interpolate(pred, feat.size()[2:], mode='bilinear', align_corners=True)
        pred = pred + self.refine(pred)
        query = pred.view(batch_size, self.num_classes, -1).permute(0, 2, 1)
        value = self.psp(self.f_value(feat)).permute(0, 2, 1)
        key = self.psp(pred)

        sim_map = torch.matmul(query, key)
        sim_map = (self.num_classes ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *feat.size()[2:])
        context = self.W(context) + feat

        return context

if __name__ == '__main__':
    x = torch.randn(2, 32, 64, 64)
    p = torch.randn(2, 19, 64, 64)
    net = CGANB(32, 32)
    y = net(x, p)
    print(y.size())



