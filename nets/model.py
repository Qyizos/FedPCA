import time
import torch
from torch import nn
import torchvision
from torchvision import models, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import time
import random
import cv2
from math import sin, cos
from einops import rearrange, repeat
from nets.basenet import *

class dilated_conv(nn.Module):
    """ same as original conv if dilation equals to 1 """
    def __init__(self, in_channel, out_channel, kernel_size=3, dropout_rate=0.0, activation=F.relu, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channel)
        self.activation = activation
        if dropout_rate > 0:
            self.drop = nn.Dropout2d(p=dropout_rate)
        else:
            self.drop = lambda x: x  # no-op

    def forward(self, x):
        # CAB: conv -> activation -> batch normal
        x = self.norm(self.activation(self.conv(x)))
        x = self.drop(x)
        return x
class ConvDownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.conv1 = dilated_conv(in_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x), x

class ConvUpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate=0.0, dilation=1):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, 2, stride=2)
        self.conv1 = dilated_conv(in_channel // 2 + out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)
        self.conv2 = dilated_conv(out_channel, out_channel, dropout_rate=dropout_rate, dilation=dilation)

    def forward(self, x, x_skip):
        x = self.up(x)
        H_diff = x.shape[2] - x_skip.shape[2]
        W_diff = x.shape[3] - x_skip.shape[3]
        x_skip = F.pad(x_skip, (0, W_diff, 0, H_diff), mode='reflect')

        x = torch.cat([x, x_skip], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

def gauss(kernel_size=3, sigma=1.5):
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    if sigma <= 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8

    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center

            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / 2 * s)
            sum_val = sum_val + kernel[i, j]

    kernel = kernel / sum_val

    return kernel

class GaussianBlurConv(nn.Module):
    def __init__(self, in_channels, kernel_size, sigma, padding):
        super(GaussianBlurConv, self).__init__()
        self.channels = in_channels
        self.padding = padding
        kernel = gauss(kernel_size, sigma)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)
        return x

class SFusion5(nn.Module):
    def __init__(self, in_dim=512, r=4):
        super(SFusion5, self).__init__()
        self.chanel_in = in_dim

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Sequential(nn.Conv2d(in_dim, 1, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(1, 1, 1, bias=False)
        )
        self.fc2 = nn.Sequential(nn.Conv2d(in_dim//2, 1, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(1, 1, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

        self.local_att1a = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )
        self.local_att1b = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )
        self.local_att1c = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )
        self.local_att1d = nn.Sequential(
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim//2),
            nn.ReLU(inplace=True)
        )
        self.local_att1e = nn.Sequential(
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim//2),
            nn.ReLU(inplace=True)
        )
        self.local_att1f = nn.Sequential(
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim//2),
            nn.ReLU(inplace=True)
        )
        self.local_att3x = nn.Sequential(
            nn.Conv2d(in_dim*2, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
        )
        self.local_att3d = nn.Sequential(
            nn.Conv2d(in_dim//2, in_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
        )

    def forward(self, x1, x2, y1, y2):
        '''
            x1, y1        # [8, 512, 7, 7]
            x2, y2        # [8, 256, 14, 14]
        '''

        stage4_x = self.local_att1a(y1)                         # [8, 512, 7, 7]
        stage4_y = self.fc1(x1)                                 # [8, 1, 7, 7]
        stage4_map = self.softmax(stage4_y)                     # [8, 1, 7, 7]
        stage4_out = self.local_att1b(stage4_x * stage4_map)    # [8, 512, 7, 7]
        stage4_out = stage4_out + x1                            # [8, 512, 7, 7]
        stage4_out = self.local_att1c(stage4_out)               # [8, 512, 7, 7]

        stage5_x = self.local_att1d(y2)                         # [8, 256, 14, 14]
        stage5_y = self.fc2(x2)                                 # [8, 1, 14, 14]
        stage5_map = self.softmax(stage5_y)                     # [8, 1, 14, 14]
        stage5_out = self.local_att1e(stage5_x * stage5_map)    # [8, 256, 14, 14]
        stage5_out = stage5_out + x2                            # [8, 256, 14, 14]
        stage5_out = self.local_att1f(stage5_out)               # [8, 256, 14, 14]

        stage5_out = self.local_att3d(stage5_out)               # [8, 512, 7, 7]
        out = self.local_att3x(torch.cat([stage4_out, stage5_out], dim=1))    # [8, 512, 7, 7]

        return out

class ResUNet34_concat5L_proFsV4k_OGC3_OSb2_fs5(nn.Module):
    def __init__(self, out_c=2, pretrained=True, fixed_feature=False):
        super(ResUNet34_concat5L_proFsV4k_OGC3_OSb2_fs5, self).__init__()
        self.resnet = models.resnet34(pretrained=pretrained)
        if fixed_feature:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.resneta = models.resnet34(pretrained=pretrained)
        if fixed_feature:
            for param in self.resneta.parameters():
                param.requires_grad = False

        self.conv1x1 = nn.Conv2d(1024, 512, kernel_size=1)

        self.GaussianBC1 = GaussianBlurConv(64, 7, 1.5, 3)
        self.sobal = Sobelxy(64)
        self.local1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
        )

        self.MidFusion = SFusion5(512)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.conv1x1a = nn.Conv2d(128, 64, kernel_size=1)

        l = [64, 64, 128, 256, 512]
        self.u5 = ConvUpBlock(l[4], l[3], dropout_rate=0.1)
        self.u6 = ConvUpBlock(l[3], l[2], dropout_rate=0.1)
        self.u7 = ConvUpBlock(l[2], l[1], dropout_rate=0.1)
        self.u8 = ConvUpBlock(l[1], l[0], dropout_rate=0.1)

        self.ce = nn.ConvTranspose2d(l[0], out_c, 2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=128,out_channels=64, kernel_size=1)
        self.conv11 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_1 = nn.BatchNorm2d(64)
        self.bn_2 = nn.BatchNorm2d(64)

    def forward(self, x, xa):

        # ----------------- Encoder 1_a -------------------
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        # ----------------- Encoder 2_a -------------------
        xa = self.resneta.conv1(xa)
        xa = self.resneta.bn1(xa)
        xa = self.resneta.relu(xa)

        x_g = self.GaussianBC1(x)
        xa_s = self.sobal(xa)
        x_rst = self.conv1x1a(torch.cat([x_g, xa_s], 1))
        x_rst = self.local1(x_rst)
        wei = self.softmax(x_rst)
        x_out1 = c1 = x_g * wei
        x_out2 = xa_s * wei

        # ----------------- Encoder 1_b -------------------
        x = self.resnet.maxpool(x_out1)
        x = c2 = self.resnet.layer1(x)
        x = c3 = self.resnet.layer2(x)
        x = c4 = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        # ----------------- Encoder 2_b -------------------
        xa = self.resneta.maxpool(x_out2)
        xa = self.resneta.layer1(xa)
        xa = self.resneta.layer2(xa)
        xa = xa4 = self.resneta.layer3(xa)
        xa = self.resneta.layer4(xa)

        # ----------------- Feature fusion -------------------
        x = self.MidFusion(x, c4, xa, xa4)
        # ----------------- Decoder -------------------
        x = self.u5(x, c4)
        x = self.u6(x, c3)
        x = self.u7(x, c2)
        x = self.u8(x, c1)
        output = self.ce(x)

        return output
