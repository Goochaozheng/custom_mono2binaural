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


def create_upconv(input_nc, output_nc, batch_norm=True, dropout=True, outermost=False):
    model = [nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)]
    if dropout:
        model.append(nn.Dropout2d(0.3, True))
    if batch_norm:
        model.append(nn.BatchNorm2d(output_nc))
    if not outermost:
        model.append(nn.LeakyReLU(0.2, True))
    else:
        model.append(nn.Sigmoid())
    return nn.Sequential(*model)
        
def create_conv(input_channels, output_channels, kernel=4, paddings=1, stride=2, batch_norm=True, dropout=True, Relu=True):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride=stride, padding=paddings)]
    if dropout:
        model.append(nn.Dropout2d(0.3, True))
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
    if(Relu):
        model.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*model)

def create_conv_3d(input_channels, output_channels, kernel=(2,4,4), paddings=(0,1,1), stride=(1,2,2), batch_norm=True, dropout=True, Relu=True):
    model = [nn.Conv3d(input_channels, output_channels, kernel, stride=stride, padding=paddings)]
    if dropout:
        model.append(nn.Dropout3d(0.3, True))
    if(batch_norm):
        model.append(nn.BatchNorm3d(output_channels))
    if(Relu):
        model.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*model)


def weights_init(m):
    #initialize weights normal_(mean, std)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight)

# 3d conv
class VisualNet(nn.Module):
    def __init__(self):
        super(VisualNet, self).__init__()
        self.visual_conv1 = create_conv_3d(3, 8)
        self.visual_conv2 = create_conv_3d(8, 16)
        self.visual_conv3 = create_conv(16, 32)
        self.visual_conv4 = create_conv(32, 64)
        self.visual_conv5 = create_conv(64, 128)
        self.visual_conv6 = create_conv(128, 128)
        # self.conv1x1 = create_conv(512, 8, 1, 0)

    def forward(self, frames):
        visual_feature = self.visual_conv1(frames)
        visual_feature = self.visual_conv2(visual_feature)
        visual_feature = visual_feature.squeeze(2)
        visual_feature = self.visual_conv3(visual_feature)
        visual_feature = self.visual_conv4(visual_feature)
        visual_feature = self.visual_conv5(visual_feature)
        visual_feature = self.visual_conv6(visual_feature)
        
        return visual_feature

# U-Net
class AudioNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioNet, self).__init__()
        #initialize layers
        self.audionet_convlayer1 = create_conv(input_nc, ngf)
        self.audionet_convlayer2 = create_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = create_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = create_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = create_conv(ngf * 8, ngf * 8)

        self.audionet_upconvlayer1 = create_upconv(1536, ngf * 8)
        self.audionet_upconvlayer2 = create_upconv(ngf * 8, ngf *4)
        self.audionet_upconvlayer3 = create_upconv(ngf * 4, ngf * 2)
        self.audionet_upconvlayer4 = create_upconv(ngf * 2, ngf)
        self.audionet_upconvlayer5 = create_upconv(ngf, output_nc, True) #outermost layer use a sigmoid to bound the mask
         #reduce dimension of extracted visual features


    def forward(self, x, visual_feat):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        visual_feat = visual_feat.view(visual_feat.shape[0], -1, 1, 1) #flatten visual feature
        visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1]) #tile visual feature
        
        audioVisual_feature = torch.cat((visual_feat, audio_conv5feature), dim=1)
        
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(audio_upconv1feature)
        audio_upconv3feature = self.audionet_upconvlayer3(audio_upconv2feature)
        audio_upconv4feature = self.audionet_upconvlayer4(audio_upconv3feature)
        mask_prediction = self.audionet_upconvlayer5(audio_upconv4feature) * 2 - 1

        return mask_prediction
