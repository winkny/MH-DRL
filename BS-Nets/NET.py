import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import torch.optim as optim
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F


class BAM(nn.Module):
    def __init__(self, nbands):
        super(BAM, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(nbands, 64, (3, 3), 1, 0),
                                   nn.ReLU(True))

        self.fc1 = nn.Sequential(nn.Linear(64, 128),
                                 nn.ReLU(True))

        self.fc2 = nn.Sequential(nn.Linear(128, nbands),
                                 nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        # print(x.shape)
        x = F.avg_pool2d(x, x.size()[2:4])
        # print(x.shape)
        x = x.view(-1, 64)
        # print(x.shape)
        x = self.fc1(x)

        x = self.fc2(x)
        # print(x.shape)
        return x.unsqueeze(2).unsqueeze(3)


class RecNet(nn.Module):
    def __init__(self):
        super(RecNet, self).__init__()
        self.conv3d_1 = nn.Sequential(nn.Conv3d(1, 24, (24, 3, 3), 1),
                                      nn.BatchNorm3d(24),
                                      nn.PReLU())

        self.conv3d_2 = nn.Sequential(nn.Conv3d(24, 48, (24, 3, 3), 1),
                                      nn.BatchNorm3d(48),
                                      nn.PReLU())

        self.pool3d = nn.MaxPool3d((18, 1, 1), (18, 1, 1))

        self.deconv3d_1 = nn.Sequential(nn.ConvTranspose3d(48, 24, (9, 3, 3), (22, 1, 1)),
                                        nn.BatchNorm3d(24),
                                        nn.PReLU())

        self.deconv3d_2 = nn.Sequential(nn.ConvTranspose3d(24, 1, (42, 3, 3), (1, 1, 1)),
                                        nn.BatchNorm3d(1))

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)

        x = self.pool3d(x)

        x = self.deconv3d_1(x)

        x = self.deconv3d_2(x)

        return x.squeeze(1)


class RecNet_103(nn.Module):
    def __init__(self):
        super(RecNet_103, self).__init__()
        self.conv3d_1 = nn.Sequential(nn.Conv3d(1, 24, (24, 3, 3), 1),
                                      nn.BatchNorm3d(24),
                                      nn.PReLU())

        self.conv3d_2 = nn.Sequential(nn.Conv3d(24, 48, (24, 3, 3), 1),
                                      nn.BatchNorm3d(48),
                                      nn.PReLU())

        self.pool3d = nn.MaxPool3d((18, 1, 1), (18, 1, 1))

        self.deconv3d_1 = nn.Sequential(nn.ConvTranspose3d(48, 24, (22, 3, 3), (22, 1, 1)),
                                        nn.BatchNorm3d(24),
                                        nn.PReLU())

        self.deconv3d_2 = nn.Sequential(nn.ConvTranspose3d(24, 1, (38, 3, 3), (1, 1, 1)),
                                        nn.BatchNorm3d(1))

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)

        x = self.pool3d(x)

        x = self.deconv3d_1(x)
        x = self.deconv3d_2(x)

        return x.squeeze(1)


class RecNet_144(nn.Module):
    def __init__(self):
        super(RecNet_144, self).__init__()
        self.conv3d_1 = nn.Sequential(nn.Conv3d(1, 24, (24, 3, 3), 1),
                                      nn.BatchNorm3d(24),
                                      nn.PReLU())

        self.conv3d_2 = nn.Sequential(nn.Conv3d(24, 48, (24, 3, 3), 1),
                                      nn.BatchNorm3d(48),
                                      nn.PReLU())

        self.pool3d = nn.MaxPool3d((18, 1, 1), (18, 1, 1))

        self.deconv3d_1 = nn.Sequential(nn.ConvTranspose3d(48, 24, (22, 3, 3), (22, 1, 1)),
                                        nn.BatchNorm3d(24),
                                        nn.PReLU())

        self.deconv3d_2 = nn.Sequential(nn.ConvTranspose3d(24, 1, (35, 3, 3), (1, 1, 1)),
                                        nn.BatchNorm3d(1))

    def forward(self, x):
        x = self.conv3d_1(x)
        x = self.conv3d_2(x)

        x = self.pool3d(x)

        x = self.deconv3d_1(x)
        x = self.deconv3d_2(x)

        return x.squeeze(1)


class BSNET_Conv(nn.Module):

    def __init__(self, nbands):
        super(BSNET_Conv, self).__init__()
        self.BAM = BAM(nbands)
        if nbands == 204:
            self.RecNet = RecNet()
        elif nbands == 103:
            self.RecNet = RecNet_103()
        elif nbands == 144:
            self.RecNet = RecNet_144()

    def forward(self, x):
        # print('before bam ', x.shape)
        BRW = self.BAM(x)

        x = x * BRW

        # print('after bam ',x.shape)

        x = x.unsqueeze(1)
        # print(x.shape)
        ret = self.RecNet(x)
        # print('after reconstruction', ret.shape)
        return ret, BRW
