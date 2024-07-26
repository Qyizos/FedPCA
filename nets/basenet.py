import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from nets import resnet as resnet

up_kwargs = {'mode': 'bilinear', 'align_corners': True}

class BaseNet(nn.Module):
    def __init__(self, nclass, backbone, aux, se_loss, dilated=True, norm_layer=None,
                 base_size=576, crop_size=608, mean=[.485, .456, .406],
                 std=[.229, .224, .225], multi_grid=False, multi_dilation=None):
        super(BaseNet, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.se_loss = se_loss
        self.mean = mean
        self.std = std
        self.base_size = base_size
        self.crop_size = crop_size
        if backbone == 'resnet50':
            self.models = resnet.resnet50(dilated=dilated,
                                              norm_layer=norm_layer, multi_grid=multi_grid,
                                              multi_dilation=multi_dilation)
        elif backbone == 'resnet101':
            self.models = resnet.resnet101(dilated=dilated,
                                               norm_layer=norm_layer, multi_grid=multi_grid,
                                               multi_dilation=multi_dilation)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        self._up_kwargs = up_kwargs

    def base_forward(self, x):
        x = self.models.conv1(x)
        x = self.models.bn1(x)
        x = self.models.relu(x)
        x = self.models.maxpool(x)
        c1 = self.models.layer1(x)
        c2 = self.models.layer2(c1)
        c3 = self.models.layer3(c2)
        c4 = self.models.layer4(c3)
        return c1, c2, c3, c4

class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        x = self.conv1(x)
        return x


