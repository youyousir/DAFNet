import torch
import torch.nn as nn
from CGAN.resnet import Resnet18

import torch.nn.functional as F
from CGAN.init_func import group_weight
from DeformCN.modules.deform_conv import DeformableAlign


def BatchNorm2d(out_chan):
#     return nn.BatchNorm2d(out_chan)
    return nn.SyncBatchNorm.convert_sync_batchnorm(nn.BatchNorm2d(out_chan))


class ConvBN(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, groups=1, *args, **kwargs):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding, groups=groups, bias=False)
        self.bn = BatchNorm2d(out_chan)
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
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride, padding=padding,groups=groups, bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)



class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    
    
class DAF(nn.Module):
    def __init__(self, low_channel, high_channel):
        super(DAF, self).__init__()
        self.conv0 = ConvBN(low_channel, low_channel, ks=1, padding=0)
        self.conv1 = ConvBN(high_channel, low_channel, ks=1, padding=0)
        self.conv2 = DeformableAlign(low_channel, low_channel, kernel_size=3, padding=1, stride=1, deformable_groups=1)
        self.bn = BatchNorm2d(low_channel)
        self.relu = nn.ReLU(True)

    def forward(self, low_feat, high_feat):
        high_feat = self.conv1(high_feat)
        high_feat = F.interpolate(high_feat, low_feat.size()[2:], mode='nearest')
        low_feat = self.conv0(low_feat)
        daf  = self.conv2(low_feat, high_feat)
        daf = self.relu(self.bn(daf))

        return daf
        
        
    
class Decoder(nn.Module):
    def __init__(self, low_channel, high_channel):
        super(Decoder1, self).__init__()
        self.conv0 = ConvBN(low_channel, low_channel, ks=1, padding=0)
        self.conv1 = ConvBN(high_channel, low_channel, ks=1, padding=0)
        self.conv2 = ConvBNReLU(low_channel, low_channel, ks=3, padding=1)

    def forward(self, low_feat, high_feat):
        high_feat = self.conv1(high_feat)
        high_feat = F.interpolate(high_feat, low_feat.size()[2:], mode='nearest')
        fuse_feat = self.conv0(low_feat) + high_feat
        fuse_feat = self.conv2(fuse_feat)

        return fuse_feat

    
    
class DAFNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(DAFNet, self).__init__()
        self.backbone = Resnet18()
        self.decoder1 = DAF(256, 512)
        self.decoder2 = DAF(128, 256)
        
        
        self.outhead1 = BiSeNetOutput(256, 64, n_classes=n_classes)
        self.outhead2 = BiSeNetOutput(512, 64, n_classes=n_classes)
        self.outhead3 = BiSeNetOutput(128, 64, n_classes=n_classes)


    def forward(self, x):
        y = self.backbone.conv1(x)
        y = self.backbone.bn1(y)
        y = self.backbone.maxpool(y)
        feat4 = self.backbone.layer1(y)
        feat8 = self.backbone.layer2(feat4)
        feat16 = self.backbone.layer3(feat8)
        feat32 = self.backbone.layer4(feat16) 

        u_feat16 = self.decoder1(feat16, feat32)
        u_feat8 = self.decoder2(feat8, u_feat16)

        coarse = self.outhead1(feat16)
        fine = self.outhead2(feat32)
        fine2 = self.outhead3(u_feat8)
        
                
        coarse = F.interpolate(coarse, x.size()[2:], mode='bilinear', align_corners=True)
        fine = F.interpolate(fine, x.size()[2:], mode='bilinear', align_corners=True)
        fine2 = F.interpolate(fine2, x.size()[2:], mode='bilinear', align_corners=True)
            
        return fine2, fine, coarse

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
    
    
    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        wd_params, nowd_params = group_weight(wd_params, nowd_params, self.backbone)
        wd_params, nowd_params = group_weight(wd_params, nowd_params, self.decoder1)
        wd_params, nowd_params = group_weight(wd_params, nowd_params, self.decoder2)
        

        lr_mul_wd_params, lr_mul_nowd_params = group_weight(lr_mul_wd_params, lr_mul_nowd_params, self.outhead1)
        lr_mul_wd_params, lr_mul_nowd_params = group_weight(lr_mul_wd_params, lr_mul_nowd_params, self.outhead2)
        lr_mul_wd_params, lr_mul_nowd_params = group_weight(lr_mul_wd_params, lr_mul_nowd_params, self.outhead3)
        
        assert len(list(self.parameters())) == len(wd_params) + len(nowd_params) + len(lr_mul_wd_params) + len(lr_mul_nowd_params)

        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

    
    
    

    
    
    
    