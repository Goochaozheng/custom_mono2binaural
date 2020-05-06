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
    dropout = nn.Dropout2d(p=0.5)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, dropout, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    dropout = nn.Dropout2d(p=0.5)
    uprelu = nn.ReLU(inplace=True)
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
        model.append(nn.ReLU(inplace=True))
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


def merge_visual_feature(visual_feature_left, visual_feature_right):
    shape = visual_feature_left.shape
    l = visual_feature_left[:,0,:,:].unsqueeze(1)
    r = visual_feature_right[:,0,:,:].unsqueeze(1)
    output = torch.cat((l,r), dim=1)

    for i in range(1, shape[1]):
        l = visual_feature_left[:,i,:,:].unsqueeze(1)
        r = visual_feature_right[:,i,:,:].unsqueeze(1)
        temp = torch.cat((l,r), dim=1)
        output = torch.cat((output, temp), dim=1)

    return output

# ResNet-18
class VisualNet(nn.Module):
    def __init__(self):
        super(VisualNet, self).__init__()
        # original_resnet = torchvision.models.resnet18(pretrained=True)
        # layers = list(original_resnet.children())[0:-2]
        # self.feature_extraction = nn.Sequential(*layers) #features before conv1x1



    def forward(self, x):
        x = self.feature_extraction(x)
        return x

# U-Net
class AudioNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioNet, self).__init__()

        self.visual_conv = create_conv(512, 128, kernel=2, stride=2, paddings=0)
        self.visual_fusion = create_conv(1024, 512, kernel=1, stride=1, paddings=0)

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)

        self.audionet_upconvlayer1 = unet_upconv(1024, ngf * 8) 
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask
        
    def forward(self, audio_spec, visual_feature_left, visual_feature_right):
        audio_conv1feature = self.audionet_convlayer1(audio_spec)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        visual_feature_left = self.visual_conv(visual_feature_left)
        visual_feature_right = self.visual_conv(visual_feature_right)
        visual_feature_left = visual_feature_left.view(visual_feature_left.shape[0], -1 ,1, 1)
        visual_feature_right = visual_feature_right.view(visual_feature_right.shape[0], -1 ,1, 1)
        visual_feature = torch.cat((visual_feature_left, visual_feature_right), dim=1)
        visual_feature = visual_feature.repeat(1,1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1])
        visual_feature = self.visual_fusion(visual_feature)
        
        audioVisual_feature = torch.cat((visual_feature, audio_conv5feature), dim=1)
        
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        mask_prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1)) * 2 - 1
        return mask_prediction
