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

def conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    dropout = nn.Dropout2d(p=0.5)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, dropout, downrelu])

def upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    dropout = nn.Dropout2d(p=0.5)
    uprelu = nn.ReLU(inplace=True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, dropout, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])

def upsample(scale=2):
    upsample = nn.Upsample(scale_factor=2)
    uprelu = nn.ReLU(inplace=True)
    return nn.Sequential(*[upsample, uprelu])


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


class AttentionNet(nn.Module):
    def __init__(self):
        super(AttentionNet, self).__init__()
        original_resnet = torchvision.models.resnet18(pretrained=True)
        layers = list(original_resnet.children())[0:-2]
        self.feature_extraction = nn.Sequential(*layers) #(512,4,8)
        
        self.mask_upsample1 = upconv(512, 128)
        self.mask_upsample2 = upsample()
        self.mask_upsample3 = upconv(128, 32)
        self.mask_upsample4 = upsample()
        self.mask_upsample5 = upconv(32, 4)
        self.mask_upsample6 = upconv(4, 1, outermost=True)
        #weight init
        self.mask_upsample1.apply(weights_init)
        self.mask_upsample2.apply(weights_init)
        self.mask_upsample3.apply(weights_init)
        self.mask_upsample4.apply(weights_init)
        self.mask_upsample5.apply(weights_init)
        self.mask_upsample6.apply(weights_init)

    def forward(self, frame):

        frame = self.feature_extraction(frame)
        attention_mask = self.mask_upsample1(frame)
        attention_mask = self.mask_upsample2(attention_mask)
        attention_mask = self.mask_upsample3(attention_mask)
        attention_mask = self.mask_upsample4(attention_mask)
        attention_mask = self.mask_upsample5(attention_mask)
        attention_mask = self.mask_upsample6(attention_mask)
        #output mask (, 1, 128, 256)
        return attention_mask



class VisualNet(nn.Module):
    def __init__(self):
        super(VisualNet, self).__init__()

        self.visual_conv1 = conv(3, 32)
        self.visual_conv2 = conv(32, 64)
        self.visual_conv3 = conv(64, 128)
        self.visual_conv4 = conv(128, 256)
        self.visual_conv5 = conv(256, 512)
        self.visual_reshape = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(2,1), stride=(2,1), padding=0)
        #weight init
        self.visual_conv1.apply(weights_init)
        self.visual_conv2.apply(weights_init)
        self.visual_conv3.apply(weights_init)
        self.visual_conv4.apply(weights_init)
        self.visual_conv5.apply(weights_init)
        self.visual_reshape.apply(weights_init)

    def forward(self, frame):

        visual_feature = self.visual_conv1(frame)
        visual_feature = self.visual_conv2(visual_feature)
        visual_feature = self.visual_conv3(visual_feature)
        visual_feature = self.visual_conv4(visual_feature)
        visual_feature = self.visual_conv5(visual_feature)
        visual_feature = self.visual_reshape(visual_feature)
        #output feeature (, 512, 8, 2)
        return visual_feature



class AudioNet(nn.Module):
    def __init__(self):
        super(AudioNet, self).__init__()
        #audio encode layer
        self.audio_conv1 = conv(2, 32) 
        self.audio_conv2 = conv(32, 64)
        self.audio_conv3 = conv(64, 128)
        self.audio_conv4 = conv(128, 256)
        self.audio_conv5 = conv(256, 512)
        #weight init
        self.audio_conv1.apply(weights_init)
        self.audio_conv2.apply(weights_init)
        self.audio_conv3.apply(weights_init)
        self.audio_conv4.apply(weights_init)
        self.audio_conv5.apply(weights_init)
        

    def forward(self, audio_spec):
        
        audio_feature = self.audio_conv1(audio_spec)
        audio_feature = self.audio_conv2(audio_feature)
        audio_feature = self.audio_conv3(audio_feature)
        audio_feature = self.audio_conv4(audio_feature)
        audio_feature = self.audio_conv5(audio_feature)
        #output audio feature (, 512, 8, 2)
        return audio_feature


class GenerationNet(nn.Module):
    def __init__(self):
        super(GenerationNet, self).__init__()

        self.audio_upconv1 = upconv(1024, 512)
        self.audio_upconv2 = upconv(512, 256)
        self.audio_upconv3 = upconv(256, 128)
        self.audio_upconv4 = upconv(128, 64)
        self.audio_upconv5 = upconv(64, 2)
        #weight init
        self.audio_upconv1.apply(weights_init)
        self.audio_upconv2.apply(weights_init)
        self.audio_upconv3.apply(weights_init)
        self.audio_upconv4.apply(weights_init)
        self.audio_upconv5.apply(weights_init)

    def forward(self, audio_visual_feature):

        mask_prediction = self.audio_upconv1(audio_visual_feature)
        mask_prediction = self.audio_upconv2(mask_prediction)
        mask_prediction = self.audio_upconv3(mask_prediction)
        mask_prediction = self.audio_upconv4(mask_prediction)
        mask_prediction = self.audio_upconv5(mask_prediction)
        #output mask (, 2, 256, 64)
        return mask_prediction
