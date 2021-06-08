#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _pair

from functions.deform_conv_func import DeformConvFunction

class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, groups=1, deformable_groups=1, im2col_step=64, bias=True):
        super(DeformConv, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels {} must be divisible by groups {}'.format(in_channels, groups))
        if out_channels % groups != 0:
            raise ValueError('out_channels {} must be divisible by groups {}'.format(out_channels, groups))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        self.im2col_step = im2col_step
        self.use_bias = bias
        
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels//groups, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()
        if not self.use_bias:
            self.bias.requires_grad = False

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    @torch.jit.script_method
    def forward(self, input, offset):
        assert 2 * self.deformable_groups * self.kernel_size[0] * self.kernel_size[1] == \
            offset.shape[1]
        return DeformConvFunction.apply(input, offset,
                                                   self.weight,
                                                   self.bias,
                                                   self.stride,
                                                   self.padding,
                                                   self.dilation,
                                                   self.groups,
                                                   self.deformable_groups,
                                                   self.im2col_step)

_DeformConv = DeformConvFunction.apply


class DeformConvPack(DeformConv):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=1, bias=True, lr_mult=0.05):
        super(DeformConvPack, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step, bias)

        out_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(self.in_channels,
                                          out_channels,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=True)
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()

    def forward(self, input):
        offset = self.conv_offset(input)
        return DeformConvFunction.apply(input, offset, 
                                          self.weight, 
                                          self.bias, 
                                          self.stride, 
                                          self.padding, 
                                          self.dilation, 
                                          self.groups,
                                          self.deformable_groups,
                                          self.im2col_step)


    
class DeformableAlign(DeformConv):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=1, bias=True, lr_mult=0.05):
        super(DeformableAlign, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step, bias)

        out_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(self.in_channels * 2,
                                     out_channels,
                                     kernel_size=(3, 3),
                                     stride=1,
                                     padding=1,
                                     bias=True)

#         self.convfusion = nn.Sequential(nn.Conv2d(self.in_channels*2, self.in_channels, 1, bias=False), 
#                                         nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=3, groups=self.in_channels,dilation=3, bias=False), 
#                                         nn.BatchNorm2d(self.in_channels))
#         self.convfusion1 = nn.Sequential(nn.Conv2d(self.in_channels*2, self.in_channels, kernel_size=3, padding=1, groups=2, bias=False),
#                                          nn.BatchNorm2d(self.in_channels),nn.ReLU(True))
#         self.convfusion2 = nn.Sequential(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, padding=0, groups=2, bias=True))
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()


    def forward(self, low_feat, high_feat):
        low_feat_origin = low_feat
        low_feat = self.conv(low_feat)
        offsetfeat = torch.cat([low_feat, high_feat], dim=1)
#         offsetfeat = self.convfusion(offsetfeat)
#         offsetfeat = self.convfusion2(offsetfeat) + offsetfeat
        offset = self.conv_offset(offsetfeat)
        warpfeat = DeformConvFunction.apply(high_feat, offset,
                                            self.weight,
                                            self.bias,
                                            self.stride,
                                            self.padding,
                                            self.dilation,
                                            self.groups,
                                            self.deformable_groups,
                                            self.im2col_step
                                            )
        feat = low_feat_origin + warpfeat
        return feat

    

class DeformableAlign1(DeformConv):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=1, bias=True, lr_mult=0.05):
        super(DeformableAlign1, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, groups, deformable_groups, im2col_step, bias)

        out_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(self.in_channels * 2,
                                     out_channels,
                                     kernel_size=(3, 3),
                                     stride=1,
                                     padding=1,
                                     bias=True)

#         self.convfusion = nn.Sequential(nn.Conv2d(self.in_channels*2, self.in_channels, 1, bias=False), 
#                                         nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=3, groups=self.in_channels,dilation=3, bias=False), 
#                                         nn.BatchNorm2d(self.in_channels))
#         self.convfusion1 = nn.Sequential(nn.Conv2d(self.in_channels*2, self.in_channels, kernel_size=3, padding=1, groups=2, bias=False),
#                                          nn.BatchNorm2d(self.in_channels),nn.ReLU(True))
#         self.convfusion2 = nn.Sequential(nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, padding=0, groups=2, bias=True))
        self.conv_offset.lr_mult = lr_mult
        self.init_offset()
#         self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)

    def init_offset(self):
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()


    def forward(self, low_feat, high_feat):
#         low_feat_origin = low_feat
#         low_feat = self.conv(low_feat)
        offsetfeat = torch.cat([low_feat, high_feat], dim=1)
#         offsetfeat = self.convfusion(offsetfeat)
#         offsetfeat = self.convfusion2(offsetfeat) + offsetfeat
        offset = self.conv_offset(offsetfeat)
        warpfeat = DeformConvFunction.apply(high_feat, offset,
                                            self.weight,
                                            self.bias,
                                            self.stride,
                                            self.padding,
                                            self.dilation,
                                            self.groups,
                                            self.deformable_groups,
                                            self.im2col_step
                                            )
        return warpfeat
    
    
    
    
    
class BiDeformableFusion(DeformConv):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, groups=1, deformable_groups=1, im2col_step=1, bias=True, lr_mult=0.05):
        super(BiDeformableFusion, self).__init__(in_channels, out_channels,
                                              kernel_size, stride, padding, dilation, groups, deformable_groups,
                                              im2col_step, bias)

        out_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset1 = nn.Conv2d(self.in_channels * 2,
                                     32,
                                     kernel_size=(3, 3),
                                     stride=self.stride,
                                     padding=1,
                                     bias=True)

        self.conv_offset2 = nn.Conv2d(32,
                                      out_channels,
                                      kernel_size=(3, 3),
                                      stride=self.stride,
                                      padding=2,
                                      dilation=2,
                                      bias=True)
        self.conv_offset3 = nn.Conv2d(32,
                                      out_channels,
                                      kernel_size=(3, 3),
                                      stride=self.stride,
                                      padding=2,
                                      dilation=2,
                                      bias=True)
        self.conv_offset1.lr_mult = lr_mult
        self.conv_offset2.lr_mult = lr_mult
        self.conv_offset3.lr_mult = lr_mult
        self.init_offset()
        # self.conv0 = nn.Conv2d(self.in_channels, self.in_channels, 1)

    def init_offset(self):
        self.conv_offset1.weight.data.zero_()
        self.conv_offset1.bias.data.zero_()
        self.conv_offset2.weight.data.zero_()
        self.conv_offset2.bias.data.zero_()
        self.conv_offset3.weight.data.zero_()
        self.conv_offset3.bias.data.zero_()

    def forward(self, feat1, feat2):
        offsetfeat = torch.cat([feat1, feat2], dim=1)
        offset = self.conv_offset1(offsetfeat)
        offset1 = self.conv_offset2(offset)
        offset2 = self.conv_offset3(offset)
        warpfeat1 = DeformConvFunction.apply(feat1, offset1,
                                            self.weight,
                                            self.bias,
                                            self.stride,
                                            self.padding,
                                            self.dilation,
                                            self.groups,
                                            self.deformable_groups,
                                            self.im2col_step)
        warpfeat2 = DeformConvFunction.apply(feat2, offset2,
                                             self.weight,
                                             self.bias,
                                             self.stride,
                                             self.padding,
                                             self.dilation,
                                             self.groups,
                                             self.deformable_groups,
                                             self.im2col_step)
        feat = warpfeat1 + warpfeat2
        return feat    

    
    