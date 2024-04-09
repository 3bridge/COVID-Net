# coding=UTF-8
import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import *
import os
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class BatchNorm2D(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, momentum=0.9, rescale=True):
        super(BatchNorm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.momentum = momentum
        self.rescale = rescale

        if (self.rescale == True):
            # define parameters gamma, beta which are learnable
            # dimension of gamma and beta should be (num_channels) ie its a one dimensional vector
            # initializing gamma as ones vector and beta as zeros vector (implies no scaling/shifting at the start)
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        # define parameters running mean and variance which is not learnable
        # keep track of mean and variance(but donot learn them), momentum is used which weighs current batch-mean and
        # variance with the running mean and variance using (momentum*runningmean+(1-momentum)*currentmean)
        self.register_buffer('runningmean', torch.zeros(num_channels))
        self.register_buffer('runningvar', torch.ones(num_channels))

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        assert len(x.shape) == 4  # 4 because (batchsize, numchannels, height, width)

        if (self.training):
            # calculate mean and variance along the dimensions other than the channel dimension
            # variance calculation is using the biased formula during training
            variance = torch.var(x, dim=[0, 2, 3], unbiased=False)
            mean = torch.mean(x, dim=[0, 2, 3])
            self.runningmean.mul_(self.momentum).add_((1 - self.momentum) * mean.detach())
            self.runningvar.mul_(self.momentum).add_((1 - self.momentum) * variance.detach())
            out = (x - mean.view([1, self.num_channels, 1, 1])) / torch.sqrt(
                variance.view([1, self.num_channels, 1, 1]) + self.epsilon)
        else:
            m = x.shape[0] * x.shape[2] * x.shape[3]
            out = (x - self.runningmean.view([1, self.num_channels, 1, 1])) / torch.sqrt(
                (m / (m - 1)) * self.runningvar.view([1, self.num_channels, 1, 1]) + self.epsilon)
            # during testing just use the running mean and (UnBiased) variance
        if (self.rescale == True):
            out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out


class BatchNormm2D(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, momentum=0.9, rescale=True):
        super(BatchNormm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.momentum = momentum
        self.rescale = rescale

        if (self.rescale == True):
            # define parameters gamma, beta which are learnable
            # dimension of gamma and beta should be (num_channels) ie its a one dimensional vector
            # initializing gamma as ones vector and beta as zeros vector (implies no scaling/shifting at the start)
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        # define parameters running mean and variance which is not learnable
        # keep track of mean and variance(but donot learn them), momentum is used which weighs current batch-mean and
        # variance with the running mean and variance using (momentum*runningmean+(1-momentum)*currentmean)
        self.register_buffer('runningmean', torch.zeros(num_channels))
        self.register_buffer('runningvar', torch.ones(num_channels))

    def forward(self, x):
        assert x.shape[1] == self.num_channels
        assert len(x.shape) == 4  # 4 because (batchsize, numchannels, height, width)

        if (self.training):
            # calculate mean and variance along the dimensions other than the channel dimension
            # variance calculation is using the biased formula during training
            variance = torch.var(x, dim=[0, 2, 3], unbiased=False)
            mean = torch.mean(x, dim=[0, 2, 3])
            self.runningmean = (1 - self.momentum) * mean + (self.momentum) * self.runningmean
            self.runningvar = (1 - self.momentum) * variance + (self.momentum) * self.runningvar
            out = (x - mean.view([1, self.num_channels, 1, 1])) / torch.sqrt(
                variance.view([1, self.num_channels, 1, 1]) + self.epsilon)
        else:
            m = x.shape[0] * x.shape[2] * x.shape[3]
            out = (x - self.runningmean.view([1, self.num_channels, 1, 1])) / torch.sqrt(
                (m / (m - 1)) * self.runningvar.view([1, self.num_channels, 1, 1]) + self.epsilon)
            # during testing just use the running mean and (UnBiased) variance
        if (self.rescale == True):
            return out

class LayerNorm2D(nn.Module):
    def __init__(self, num_channels, epsilon = 1e-5):
        super(LayerNorm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
      #  assert list(x.shape)[1] == self.num_channels
       # assert len(x.shape) == 4 # 4 because len((batchsize, numchannels, height, width)) = 4

        variance, mean = torch.var(x, dim = [1,2, 3], unbiased=False), torch.mean(x, dim = [1,2, 3])
        out = (x-mean.view([-1, 1, 1, 1]))/torch.sqrt(variance.view([-1, 1, 1, 1])+self.epsilon)

        out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out



class LayerNormm2D(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5):
        super(LayerNormm2D, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        assert list(x.shape)[1] == self.num_channels
        assert len(x.shape) == 4  # 4 because len((batchsize, numchannels, height, width)) = 4
        variance, mean = torch.var(x, dim=[1, 2, 3], unbiased=False), torch.mean(x, dim=[1, 2, 3])

        out = (x - mean.view([-1, 1, 1, 1])) / torch.sqrt(variance.view([-1, 1, 1, 1]) + self.epsilon)
        return out


class BatchChannelNorm(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, momentum=0.9):
        super(BatchChannelNorm, self).__init__()
        self.num_channels = num_channels
        self.epsilon = epsilon
        self.momentum = momentum
        # Assuming num_features is the product of H*W for LayerNorm
        self.Batchh = BatchNormm2D(self.num_channels, epsilon=self.epsilon)
        self.layeer = LayerNormm2D(self.num_channels, epsilon=self.epsilon)  # num_features should be [C, H, W]

        # The BCN variable to be learnt
        self.BCN_var = nn.Parameter(torch.ones(self.num_channels))
        # Gamma and Beta for rescaling
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        X = self.Batchh(x)
        Y = self.layeer(x)
        out = self.BCN_var.view([1, self.num_channels, 1, 1]) * X + (
                1 - self.BCN_var.view([1, self.num_channels, 1, 1])) * Y
        out = self.gamma.view([1, self.num_channels, 1, 1]) * out + self.beta.view([1, self.num_channels, 1, 1])
        return out


# 定义一个模块，用于将特征图展平
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

# 定义一个基础的卷积块
class Block(nn.Module):
    def __init__(self, n_input, n_out):
        super(Block, self).__init__()
        # 使用多个卷积层来逐步调整通道数，包括减少、增加和保持不变的操作
        self.network = nn.Sequential(nn.Conv2d(in_channels=n_input, out_channels=n_input // 2, kernel_size=1),
                                     nn.Conv2d(in_channels=n_input // 2, out_channels=int(3 * n_input / 4),
                                               kernel_size=1),
                                     nn.Conv2d(in_channels=int(3 * n_input / 4), out_channels=int(3 * n_input / 4),
                                               kernel_size=3, groups=int(3 * n_input / 4), padding=1),
                                     nn.Conv2d(in_channels=int(3 * n_input / 4), out_channels=n_input // 2,
                                               kernel_size=1),
                                     nn.Conv2d(in_channels=n_input // 2, out_channels=n_out, kernel_size=1))

    def forward(self, x):
        return self.network(x)

# 定义CovidNet模型类
class CovidNet(nn.Module):
    def __init__(self, bnd=False, bna=False, label_smoothing=False,
                 n_classes=2, hidden_size=1024, emmbedding_size=128):
        super(CovidNet, self).__init__()
        # 初始化网络中的各个卷积层和卷积块的通道数
        filters = {
            'pexp1_1': [64, 256],
            'pexp1_2': [256, 256],
            'pexp1_3': [256, 256],
            'pexp2_1': [256, 512],
            'pexp2_2': [512, 512],
            'pexp2_3': [512, 512],
            'pexp2_4': [512, 512],
            'pexp3_1': [512, 1024],
            'pexp3_2': [1024, 1024],
            'pexp3_3': [1024, 1024],
            'pexp3_4': [1024, 1024],
            'pexp3_5': [1024, 1024],
            'pexp3_6': [1024, 1024],
            'pexp4_1': [1024, 2048],
            'pexp4_2': [2048, 2048],
            'pexp4_3': [2048, 2048],
        }

        self.label_smoothing = label_smoothing
        self.bnd = bnd
        self.bna = bna
        if bnd:
            self.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3))
            self.add_module('conv1_1x1', nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1))
            self.add_module('conv2_1x1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1))
            self.add_module('conv3_1x1', nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1))
            self.add_module('conv4_1x1', nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1))

            # bn for ucsd data
            # self.add_module('bn0', nn.Sequential(nn.BatchNorm2d(64), nn.ReLU()))
            # self.add_module('bn1', nn.Sequential(nn.BatchNorm2d(256), nn.ReLU()))
            # self.add_module('gn2', nn.Sequential(GroupNorm(num_features=512), nn.ReLU()))
            # self.add_module('bn3', nn.Sequential(nn.BatchNorm2d(1024), nn.ReLU()))
            # self.add_module('bn4', nn.Sequential(nn.BatchNorm2d(2048), nn.ReLU()))
            # # bn for new data
            # self.add_module('bn0new', nn.Sequential(nn.BatchNorm2d(64), nn.ReLU()))
            # self.add_module('bn1new', nn.Sequential(nn.BatchNorm2d(256), nn.ReLU()))
            # self.add_module('gn2new', nn.Sequential(GroupNorm(num_features=512), nn.ReLU()))
            # self.add_module('bn3new', nn.Sequential(nn.BatchNorm2d(1024), nn.ReLU()))
            # self.add_module('bn4new', nn.Sequential(nn.BatchNorm2d(2048), nn.ReLU()))
            # bn for ucsd data
            self.add_module('bn0', nn.Sequential(BatchChannelNorm(num_channels=64), nn.ReLU()))
            self.add_module('bn1', nn.Sequential(BatchChannelNorm(num_channels=256), nn.ReLU()))
            self.add_module('bn2', nn.Sequential(BatchChannelNorm(num_channels=512), nn.ReLU()))
            self.add_module('bn3', nn.Sequential(BatchChannelNorm(num_channels=1024), nn.ReLU()))
            self.add_module('bn4', nn.Sequential(BatchChannelNorm(num_channels=2048), nn.ReLU()))
            # bn for new data
            self.add_module('bn0new', nn.Sequential(BatchChannelNorm(num_channels=64), nn.ReLU()))
            self.add_module('bn1new', nn.Sequential(BatchChannelNorm(num_channels=256), nn.ReLU()))
            self.add_module('bn2new', nn.Sequential(BatchChannelNorm(num_channels=512), nn.ReLU()))
            self.add_module('bn3new', nn.Sequential(BatchChannelNorm(num_channels=1024), nn.ReLU()))
            self.add_module('bn4new', nn.Sequential(BatchChannelNorm(num_channels=2048), nn.ReLU()))
        else:
            self.add_module('conv1', nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3))
            self.add_module('conv1_1x1', nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1))
            self.add_module('conv2_1x1', nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1))
            self.add_module('conv3_1x1', nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1))
            self.add_module('conv4_1x1', nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1))

        for key in filters.keys():
            self.add_module(key, Block(filters[key][0], filters[key][1]))

        self.add_module('avg', nn.AdaptiveAvgPool2d(output_size=(2, 2))) # 自适应平均池化层

        self.add_module('flat', Flatten())# 展平层
        # 定义全连接层以进行特征到类别的映射
        self.add_module('fc1', nn.Linear(2048 * 2 * 2, 1024))
        self.add_module('fc2', nn.Linear(1024, 256))

        # bn
        if bna:
            # for ucsd
            self.add_module('bn5', nn.Sequential(nn.BatchNorm1d(1024), nn.ReLU()))
            self.add_module('bn6', nn.Sequential(nn.BatchNorm1d(256), nn.ReLU()))
            # for new
            self.add_module('bn5new', nn.Sequential(nn.BatchNorm1d(1024), nn.ReLU()))
            self.add_module('bn6new', nn.Sequential(nn.BatchNorm1d(256), nn.ReLU()))
            # self.add_module('bn5', nn.Sequential(BatchChannelNorm(num_channels=1024), nn.ReLU()))
            # self.add_module('bn6', nn.Sequential(BatchChannelNorm(num_channels=256), nn.ReLU()))
            # # for new
            # self.add_module('bn5new', nn.Sequential(BatchChannelNorm(num_channels=1024), nn.ReLU()))
            # self.add_module('bn6new', nn.Sequential(BatchChannelNorm(num_channels=256), nn.ReLU()))
        # contrastive feature
        self.hidden_size = hidden_size
        self.emmbedding_size = emmbedding_size
        self.add_module('feature1', nn.Linear(2048 * 2 * 2, self.hidden_size))
        self.add_module('feature2', nn.Linear(self.hidden_size, self.emmbedding_size))

        # bn
        if bna:
            # for ucsd
            self.add_module('fbn5', nn.Sequential(nn.BatchNorm1d(self.hidden_size), nn.ReLU()))
            self.add_module('fbn6', nn.Sequential(nn.BatchNorm1d(self.emmbedding_size), nn.ReLU()))
            # for new
            self.add_module('fbn5new', nn.Sequential(nn.BatchNorm1d(self.hidden_size), nn.ReLU()))
            self.add_module('fbn6new', nn.Sequential(nn.BatchNorm1d(self.emmbedding_size), nn.ReLU()))

            # self.add_module('fbn5', nn.Sequential(BatchChannelNorm(num_channels=self.hidden_size), nn.ReLU()))
            # self.add_module('fbn6', nn.Sequential(BatchChannelNorm(num_channels=self.emmbedding_size), nn.ReLU()))
            # # for new
            # self.add_module('fbn5new', nn.Sequential(BatchChannelNorm(num_channels=self.hidden_size), nn.ReLU()))
            # self.add_module('fbn6new', nn.Sequential(BatchChannelNorm(num_channels=self.emmbedding_size), nn.ReLU()))
        classifier_list = [
            nn.Linear(256, n_classes),# 分类器的全连接层
            nn.Sigmoid() # 使用Sigmoid函数作为激活函数，用于二分类任务

        ]
        self.feature_num = 256
        self.classifier = nn.Sequential(*classifier_list)# 将分类器的各层组合成一个顺序模型

    def forward(self, x, site):
        # 定义数据的前向传播过程
        conv1 = self.conv1(x)

        if self.bnd:
            if site == 'ucsd':
                conv1 = self.bn0(conv1)
            else:
                conv1 = self.bn0new(conv1)
        x = F.max_pool2d(conv1, 2)

        out_conv1_1x1 = self.conv1_1x1(x)

        if self.bnd:
            if site == 'ucsd':
                out_conv1_1x1 = self.bn1(out_conv1_1x1)
            else:
                out_conv1_1x1 = self.bn1new(out_conv1_1x1)

        pepx11 = self.pexp1_1(x)
        pepx12 = self.pexp1_2(pepx11 + out_conv1_1x1)
        pepx13 = self.pexp1_3(pepx12 + pepx11 + out_conv1_1x1)

        conv2 = self.conv2_1x1(pepx12 + pepx11 + pepx13 + out_conv1_1x1)

        if self.bnd:
            if site == 'ucsd':
                conv2 = self.bn2(conv2)
            else:
                conv2 = self.bn2new(conv2)
        out_conv2_1x1 = F.max_pool2d(conv2, 2)

        pepx21 = self.pexp2_1(
            F.max_pool2d(pepx13, 2) + F.max_pool2d(pepx11, 2) + F.max_pool2d(pepx12, 2) + F.max_pool2d(out_conv1_1x1,
                                                                                                       2))
        pepx22 = self.pexp2_2(pepx21 + out_conv2_1x1)
        pepx23 = self.pexp2_3(pepx22 + pepx21 + out_conv2_1x1)
        pepx24 = self.pexp2_4(pepx23 + pepx21 + pepx22 + out_conv2_1x1)

        conv3 = self.conv3_1x1(pepx22 + pepx21 + pepx23 + pepx24 + out_conv2_1x1)

        if self.bnd:
            if site == 'ucsd':
                conv3 = self.bn3(conv3)
            else:
                conv3 = self.bn3new(conv3)
        out_conv3_1x1 = F.max_pool2d(conv3, 2)

        pepx31 = self.pexp3_1(
            F.max_pool2d(pepx24, 2) + F.max_pool2d(pepx21, 2) + F.max_pool2d(pepx22, 2) + F.max_pool2d(pepx23,
                                                                                                       2) + F.max_pool2d(
                out_conv2_1x1, 2))
        pepx32 = self.pexp3_2(pepx31 + out_conv3_1x1)
        pepx33 = self.pexp3_3(pepx31 + pepx32 + out_conv3_1x1)
        pepx34 = self.pexp3_4(pepx31 + pepx32 + pepx33 + out_conv3_1x1)
        pepx35 = self.pexp3_5(pepx31 + pepx32 + pepx33 + pepx34 + out_conv3_1x1)
        pepx36 = self.pexp3_6(pepx31 + pepx32 + pepx33 + pepx34 + pepx35 + out_conv3_1x1)

        conv4 = self.conv4_1x1(pepx31 + pepx32 + pepx33 + pepx34 + pepx35 + pepx36 + out_conv3_1x1)

        if self.bnd:
            if site == 'ucsd':
                conv4 = self.bn4(conv4)
            else:
                conv4 = self.bn4new(conv4)
        out_conv4_1x1 = F.max_pool2d(conv4, 2)

        pepx41 = self.pexp4_1(
            F.max_pool2d(pepx31, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx32, 2) + F.max_pool2d(pepx34,
                                                                                                       2) + F.max_pool2d(
                pepx35, 2) + F.max_pool2d(pepx36, 2) + F.max_pool2d(out_conv3_1x1, 2))
        pepx42 = self.pexp4_2(pepx41 + out_conv4_1x1)
        pepx43 = self.pexp4_3(pepx41 + pepx42 + out_conv4_1x1)

        avg = self.avg(pepx43)

        flat = self.flat(avg)

        fc1 = self.fc1(flat)

        if self.bna:
            if site == 'ucsd':
                fc1 = self.bn5(fc1)
            else:
                fc1 = self.bn5new(fc1)
        fc2 = self.fc2(fc1)

        if self.bna:
            if site == 'ucsd':
                fc2 = self.bn6(fc2)
            else:
                fc2 = self.bn6new(fc2)

        logits = self.classifier(fc2)

        # contrastive loss
        feature1 = self.feature1(flat)
        if self.bna:
            if site == 'ucsd':
                feature1 = self.fbn5(feature1)
            else:
                feature1 = self.fbn5new(feature1)
        feature2 = self.feature2(feature1)
        if self.bna:
            if site == 'ucsd':
                feature2 = self.fbn6(feature2)
            else:
                feature2 = self.fbn6new(feature2)

        return logits, feature2
