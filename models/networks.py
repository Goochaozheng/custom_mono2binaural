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
    dropout = nn.Dropout2d(p=0.5)
    uprelu = nn.LeakyReLU(0.2, True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, dropout, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])
        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
    if(Relu):
        model.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*model)

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


class VisualNet(nn.Module):
    def __init__(self):
        super(VisualNet, self).__init__()
        original_net = torchvision.models.googlenet(pretrained=True)
        layers = list(googlenet.children())[:-3]
        self.feature_extraction = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature_extraction(x)
        return x

# U-Net
class AudioNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioNet, self).__init__()
        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(768, ngf * 8) #1296 (audio-visual feature) = 256 (visual feature) + 512 (audio feature)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16 + 256, ngf *4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8 + 256, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4 + 256, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2 + 256, output_nc, True) #outermost layer use a sigmoid to bound the mask
        
        self.visual_fusion1 = create_conv(1024, 8, 1, 0) #reduce dimension of extracted visual features
        self.visual_fusion2 = create_conv(1024, 8, 1, 0) #reduce dimension of extracted visual features
        self.visual_fusion3 = create_conv(1024, 8, 1, 0) #reduce dimension of extracted visual features
        self.visual_fusion4 = create_conv(1024, 8, 1, 0) #reduce dimension of extracted visual features
        self.visual_fusion5 = create_conv(1024, 8, 1, 0) #reduce dimension of extracted visual features

    def forward(self, x, visual_feat):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        visual_feat5 = self.visual_fusion5(visual_feat)
        visual_feat5 = visual_feat5.view(visual_feat5.shape[0], -1, 1, 1) #flatten visual feature
        visual_feat5 = visual_feat5.repeat(1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1]) #tile visual feature
        audio_upconv1feature = self.audionet_upconvlayer1(torch.cat((audio_conv5feature, visual_feat5), dim=1))

        visual_feat4 = self.visual_fusion4(visual_feat)
        visual_feat4 = visual_feat4.view(visual_feat4.shape[0], -1, 1, 1) #flatten visual feature
        visual_feat4 = visual_feat4.repeat(1, 1, audio_conv4feature.shape[-2], audio_conv4feature.shape[-1]) #tile visual feature
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature, visual_feat4), dim=1))

        visual_feat3 = self.visual_fusion3(visual_feat)
        visual_feat3 = visual_feat3.view(visual_feat3.shape[0], -1, 1, 1) #flatten visual feature
        visual_feat3 = visual_feat3.repeat(1, 1, audio_conv3feature.shape[-2], audio_conv3feature.shape[-1]) #tile visual feature        
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature, visual_feat3), dim=1))

        visual_feat2 = self.visual_fusion2(visual_feat)
        visual_feat2 = visual_feat2.view(visual_feat2.shape[0], -1, 1, 1) #flatten visual feature
        visual_feat2 = visual_feat2.repeat(1, 1, audio_conv2feature.shape[-2], audio_conv2feature.shape[-1]) #tile visual feature        
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature, visual_feat2), dim=1))

        visual_feat1 = self.visual_fusion1(visual_feat)
        visual_feat1 = visual_feat1.view(visual_feat1.shape[0], -1, 1, 1) #flatten visual feature
        visual_feat1 = visual_feat1.repeat(1, 1, audio_conv1feature.shape[-2], audio_conv1feature.shape[-1]) #tile visual feature        
        mask_prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature, visual_feat1), dim=1)) * 2 - 1

        return mask_prediction
