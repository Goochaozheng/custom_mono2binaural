#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Define network components of the model

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import torchvision

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    # dropout = nn.Dropout2d(p=0.5)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    # dropout = nn.Dropout2d(p=0.5)
    uprelu = nn.ReLU(inplace=True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])
        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
    if(Relu):
        model.append(nn.ReLU(inplace=True))
    return nn.Sequential(*model)


def audio_1dconv():
    conv1d = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=200, stride=40, padding=0)
    relu =  nn.LeakyReLU(0.2, True)
    norm = nn.BatchNorm1d(128)
    return nn.Sequential(*[conv1d, relu, norm])


def weights_init(m):
    #initialize weights normal_(mean, std)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

# ResNet-18
class VisualNet(nn.Module):
    def __init__(self):
        super(VisualNet, self).__init__()
        original_resnet = torchvision.models.resnet18(pretrained=True)
        layers = list(original_resnet.children())[0:-2]
        self.feature_extraction = nn.Sequential(*layers) #features before conv1x1

    def forward(self, x):
        x = self.feature_extraction(x)
        return x

# U-Net
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.audio_1dconv = audio_1dconv()
        self.audio_conv1 = unet_conv(1, 32)
        self.audio_conv2 = unet_conv(32, 64)
        self.audio_conv3 = unet_conv(64, 128)
        self.audio_conv4 = unet_conv(128, 256)
        self.audio_conv5 = unet_conv(256, 512)

        self.gen_upconv1 = 

    def forward(self, audio_spec, visual_frame):
        

