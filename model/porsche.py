import torch
import copy
import cv2
import numpy as np
import re
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
from torch.nn import init
from torch.nn.init import xavier_uniform_
from einops import rearrange
from torch import einsum
from torch.nn.modules.container import ModuleList
# from .backbones.resnet import resnet50, resnet18
from .backbones.resnet_ccbc import *
from .backbones.attention import CrossAttentionModule
from .backbones.transmatcher import *

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool1(x)
        return x

class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool1(x)
        return x

class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class base_resblock(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resblock, self).__init__()

        model_base = resnet50(pretrained=True,last_conv_stride=1, last_conv_dilation=1)

        model_base.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.base = model_base

    def forward(self, x, idx):
        if idx not in [1,2,3,4]:
            raise NotImplementedError

        if idx == 1:
            x = self.base.layer1(x)
        elif idx == 2:
            x = self.base.layer2(x)
        elif idx == 3:
            x = self.base.layer3(x)
        else:
            x = self.base.layer4(4)

        return x

class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out
    
class LearnableWeights(nn.Module):
    def __init__(self):
        super(LearnableWeights, self).__init__()
        self.w1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.w2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        out = x1 * self.w1 + x2 * self.w2
        return out

class AdaptivePool2d(nn.Module):
    def __init__(self, output_h, output_w, pool_type='avg'):
        super(AdaptivePool2d, self).__init__()

        self.output_h = output_h
        self.output_w = output_w
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, input_h, input_w = x.shape

        if (input_h > self.output_h) or (input_w > self.output_w):
            self.stride_h = input_h // self.output_h
            self.stride_w = input_w // self.output_w
            self.kernel_size = (input_h - (self.output_h - 1) * self.stride_h, input_w - (self.output_w - 1) * self.stride_w)

            if self.pool_type == 'avg':
                y = nn.AvgPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
            else:
                y = nn.MaxPool2d(kernel_size=self.kernel_size, stride=(self.stride_h, self.stride_w), padding=0)(x)
        else:
            y = x

        return y
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//16, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//16, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x,y, atten=False):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C,height, width = x.size()
        assert x.size() == y.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(y).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(y).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize,C, height, width)

        out = self.gamma*out + x

        if atten == False:
            return out
        else:
            return out, attention

class PORSCHE(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, class_num, arch='resnet50', final_layer='layer3', neck=512,
                 nheads=1, num_encoder_layers=2, dim_feedforward=2048, 
                 dropout=0.1, pretrained=True):
        super(PORSCHE, self).__init__()
        match = re.search(r'\d+', arch)
        if match:
            depth = int(match.group())
        else:
            raise KeyError("Unsupported arch:", arch)
        self.depth = depth
        self.final_layer = final_layer
        self.neck = neck
        self.pretrained = pretrained
        
        self.visible_module = visible_module(arch=arch)
        self.thermal_module = thermal_module(arch=arch)
        self.base_module = resnet50(pretrained=pretrained)
        # self.max1 = nn.MaxPool2d(kernel_size=5, stride=5)
        # self.max2 = nn.MaxPool2d(kernel_size=3, stride=3)
        # self.max3 = nn.MaxPool2d(kernel_size=1, stride=1)
        self.num_ftrs = 2048 * 1 * 1

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs//4, neck, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(neck, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs//2, neck, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(neck, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.conv_block3 = nn.Sequential(
            BasicConv(self.num_ftrs, neck, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(neck, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        out_planes = self.num_ftrs//2
        self.bottleneck = nn.BatchNorm1d(out_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(out_planes, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        # transmatcher
        self.encoder = None
        if num_encoder_layers > 0:
            encoder_layer1 = TransformerEncoderLayer(out_planes, nheads, dim_feedforward, dropout)
            encoder_norm = None
            self.encoder = TransformerEncoder(encoder_layer1, num_encoder_layers, encoder_norm)

        self.num_features = out_planes

    def forward(self, inputs1, inputs2, modal=0, block=[0, 0, 0, 0]):
        if modal == 0:
            x1 = self.visible_module(inputs1)
            x2 = self.thermal_module(inputs2)
            x = torch.cat((x1, x2), dim=0)
        elif modal == 1:
            x = self.visible_module(inputs1)
        elif modal == 2:
            x = self.thermal_module(inputs2)
        
        x1, x2, x3, x4, x5, f1, f2, f3 = self.base_module(x, block)
        # x1: [bs*2, 64, 32, 64]
        # x2: [bs*2, 256, 32, 64]
        # x3: [bs*2, 512, 16, 32]
        # x4: [bs*2, 1024, 8, 16]
        # x5: [bs*2, 2048, 4, 8]
        # f1: [bs*2, 512, 1, 1]
        # f2: [bs*2, 1024, 1, 1]
        # f3: [bs*2, 2048, 1, 1]

        xs1 = self.conv_block1(x3)
        xs2 = self.conv_block2(x4)
        xs3 = self.conv_block3(x5)

        xs1_pool = self.avgpool(xs1)
        xs1_pool = xs1_pool.view(xs1_pool.size(0), xs1_pool.size(1))
        xs2_pool = self.avgpool(xs2)
        xs2_pool = xs2_pool.view(xs2_pool.size(0), xs2_pool.size(1))
        xs3_pool = self.avgpool(xs3)
        xs3_pool = xs3_pool.view(xs3_pool.size(0), xs3_pool.size(1))

        feat1 = self.bottleneck(xs1_pool)
        feat2 = self.bottleneck(xs2_pool)
        feat3 = self.bottleneck(xs3_pool)

        out1 = xs1.permute(0, 2, 3, 1)  # [b, h, w, c]
        out2 = xs2.permute(0, 2, 3, 1)
        out3 = xs3.permute(0, 2, 3, 1)

        if self.encoder is not None:
            b1, c1, h1, w1 = xs1.size()
            y1 = xs1.view(b1, c1, -1).permute(2, 0, 1)
            y1 = self.encoder(y1)
            y1 = y1.permute(1, 0, 2).reshape(b1, h1, w1, -1)
            out1 = torch.cat((out1, y1), dim=-1)

            b2, c2, h2, w2 = xs2.size()
            y2 = xs2.view(b2, c2, -1).permute(2, 0, 1)
            y2 = self.encoder(y2)
            y2 = y2.permute(1, 0, 2).reshape(b2, h2, w2, -1)
            out2 = torch.cat((out2, y2), dim=-1)

            b3, c3, h3, w3 = xs3.size()
            y3 = xs3.view(b3, c3, -1).permute(2, 0, 1)
            y3 = self.encoder(y3)
            y3 = y3.permute(1, 0, 2).reshape(b3, h3, w3, -1)
            out3 = torch.cat((out3, y3), dim=-1)

        if self.training:
            return out1, out2, out3, \
                xs1_pool.to(torch.float32), xs2_pool.to(torch.float32), xs3_pool.to(torch.float32),\
                self.classifier(feat1), self.classifier(feat2), self.classifier(feat3),\
                f1, f2, f3
        else:
            return out1, out2, out3, f1, f2, f3

# class PORSCHE(nn.Module):
#     __factory = {
#         18: torchvision.models.resnet18,
#         34: torchvision.models.resnet34,
#         50: torchvision.models.resnet50,
#         101: torchvision.models.resnet101,
#         152: torchvision.models.resnet152,
#     }

#     def __init__(self, class_num, arch='resnet50', final_layer='layer3', neck=512,
#                  nheads=1, num_encoder_layers=2, dim_feedforward=2048, 
#                  dropout=0.1, pretrained=True):
#         super(PORSCHE, self).__init__()
#         match = re.search(r'\d+', arch)
#         if match:
#             depth = int(match.group())
#         else:
#             raise KeyError("Unsupported arch:", arch)
#         self.depth = depth
#         self.final_layer = final_layer
#         self.neck = neck
#         self.pretrained = pretrained
        
#         self.visible_module = visible_module(arch=arch)
#         self.thermal_module = thermal_module(arch=arch)
#         self.base_module = resnet50(pretrained=pretrained)
#         # self.max1 = nn.MaxPool2d(kernel_size=5, stride=5)
#         # self.max2 = nn.MaxPool2d(kernel_size=3, stride=3)
#         # self.max3 = nn.MaxPool2d(kernel_size=1, stride=1)
#         self.num_ftrs = 2048 * 1 * 1

#         self.conv_block1 = nn.Sequential(
#             BasicConv(self.num_ftrs//4, neck, kernel_size=1, stride=1, padding=0, relu=True),
#             BasicConv(neck, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
#         )
#         self.conv_block2 = nn.Sequential(
#             BasicConv(self.num_ftrs//2, neck, kernel_size=1, stride=1, padding=0, relu=True),
#             BasicConv(neck, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
#         )
#         self.conv_block3 = nn.Sequential(
#             BasicConv(self.num_ftrs, neck, kernel_size=1, stride=1, padding=0, relu=True),
#             BasicConv(neck, self.num_ftrs//2, kernel_size=3, stride=1, padding=1, relu=True)
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

#         out_planes = self.num_ftrs//2
#         self.bottleneck = nn.BatchNorm1d(out_planes)
#         self.bottleneck.bias.requires_grad_(False)
#         self.classifier = nn.Linear(out_planes, class_num, bias=False)

#         self.bottleneck.apply(weights_init_kaiming)
#         self.classifier.apply(weights_init_classifier)
#         # transmatcher
#         self.encoder = None
#         if num_encoder_layers > 0:
#             encoder_layer1 = TransformerEncoderLayer(out_planes, nheads, dim_feedforward, dropout)
#             encoder_norm = None
#             self.encoder = TransformerEncoder(encoder_layer1, num_encoder_layers, encoder_norm)

#         self.num_features = out_planes

#     def forward(self, inputs1, inputs2=None, modal=2, block=[0, 0, 0, 0]):
#         if modal == 0:
#             x1 = self.visible_module(inputs1)
#             x2 = self.thermal_module(inputs2)
#             x = torch.cat((x1, x2), dim=0)
#         elif modal == 1:
#             x = self.visible_module(inputs1)
#         elif modal == 2:
#             x = self.thermal_module(inputs1)
        
#         x1, x2, x3, x4, x5, f1, f2, f3 = self.base_module(x, block)
#         # x1: [bs*2, 64, 32, 64]
#         # x2: [bs*2, 256, 32, 64]
#         # x3: [bs*2, 512, 16, 32]
#         # x4: [bs*2, 1024, 8, 16]
#         # x5: [bs*2, 2048, 4, 8]
#         # f1: [bs*2, 512, 1, 1]
#         # f2: [bs*2, 1024, 1, 1]
#         # f3: [bs*2, 2048, 1, 1]

#         xs1 = self.conv_block1(x3)
#         xs2 = self.conv_block2(x4)
#         xs3 = self.conv_block3(x5)

#         xs1_pool = self.avgpool(xs1)
#         xs1_pool = xs1_pool.view(xs1_pool.size(0), xs1_pool.size(1))
#         xs2_pool = self.avgpool(xs2)
#         xs2_pool = xs2_pool.view(xs2_pool.size(0), xs2_pool.size(1))
#         xs3_pool = self.avgpool(xs3)
#         xs3_pool = xs3_pool.view(xs3_pool.size(0), xs3_pool.size(1))

#         feat1 = self.bottleneck(xs1_pool)
#         feat2 = self.bottleneck(xs2_pool)
#         feat3 = self.bottleneck(xs3_pool)

#         out1 = xs1.permute(0, 2, 3, 1)  # [b, h, w, c]
#         out2 = xs2.permute(0, 2, 3, 1)
#         out3 = xs3.permute(0, 2, 3, 1)

#         if self.encoder is not None:
#             b1, c1, h1, w1 = xs1.size()
#             y1 = xs1.view(b1, c1, -1).permute(2, 0, 1)
#             y1 = self.encoder(y1)
#             y1 = y1.permute(1, 0, 2).reshape(b1, h1, w1, -1)
#             out1 = torch.cat((out1, y1), dim=-1)

#             b2, c2, h2, w2 = xs2.size()
#             y2 = xs2.view(b2, c2, -1).permute(2, 0, 1)
#             y2 = self.encoder(y2)
#             y2 = y2.permute(1, 0, 2).reshape(b2, h2, w2, -1)
#             out2 = torch.cat((out2, y2), dim=-1)

#             b3, c3, h3, w3 = xs3.size()
#             y3 = xs3.view(b3, c3, -1).permute(2, 0, 1)
#             y3 = self.encoder(y3)
#             y3 = y3.permute(1, 0, 2).reshape(b3, h3, w3, -1)
#             out3 = torch.cat((out3, y3), dim=-1)

#         return self.classifier(feat3)