import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import math
import pdb
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
import copy


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        if type == "linear":
            self.fc = nn.Linear(bottleneck_dim, class_num,bias=False)
        else:
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num),
                                 name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class ResNet_sdaE_all(nn.Module):
    def __init__(self):
        super().__init__()
        model_resnet = torchvision.models.resnet50(True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        #self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu,
                                            self.maxpool, self.layer1,
                                            self.layer2, self.layer3,
                                            self.layer4, self.avgpool)
        #self.bottle=nn.Sequential(nn.Linear(2048, 256),nn.BatchNorm1d(256))
        self.bottle = nn.Linear(2048, 512)
        self.bn = nn.BatchNorm1d(512)
        self.em = nn.Embedding(4, 512)
        self.mask = torch.empty(1, 512)

    def forward(self, x, t, s=100,all_out=False):
        out = self.feature_layers(x)
        out = out.view(out.size(0), -1)
        out = self.bottle(out)
        out = self.bn(out)
        #t=0
        if all_out==False:
            t = torch.LongTensor([t]).cuda()
            self.mask = nn.Sigmoid()(self.em(t) * s)
            flg = torch.isnan(self.mask).sum()
            if flg != 0:
                print('nan occurs')
            #print(self.mask.shape)
            out = out * self.mask
        '''if t == 1:
            t_ = torch.LongTensor([0]).cuda()
            self.mask = nn.Sigmoid()(self.em(t_) * s)
            t = torch.LongTensor([t]).cuda()
            mask = nn.Sigmoid()(self.em(t) * s)
            flg = torch.isnan(mask).sum()
            if flg != 0:
                print('nan occurs')
            #print(self.mask.shape)
            out = out * mask'''
        if all_out:
            t0=torch.LongTensor([0]).cuda()
            t1=torch.LongTensor([1]).cuda()
            t2=torch.LongTensor([2]).cuda()
            t3=torch.LongTensor([3]).cuda()
            mask0=nn.Sigmoid()(self.em(t0) * 100)
            mask1=nn.Sigmoid()(self.em(t1) * 100)
            mask2=nn.Sigmoid()(self.em(t2) * 100)
            mask3=nn.Sigmoid()(self.em(t3) * 100)
            self.mask=mask0
            #print(mask0.shape,out.shape)
            out0 = out * mask0
            out1 = out * mask1
            out2 = out * mask2
            out3 = out * mask3

        if all_out:
            return (out0,out1,out2,out3), (self.mask,mask1,mask2,mask3)
        else:
            return out, self.mask
