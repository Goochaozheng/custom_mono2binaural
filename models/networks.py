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
class AudioNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioNet, self).__init__()
        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)

        resnet = torchvision.models.resnet18(pretrained=True)
        layers = list(resnet.children())

        #Resnet block
        self.residual_block1 = torch.nn.Sequential(*layers[:5]) #(, 64, 32, 64)
        self.residual_block2 = layers[5] #(, 128, 16, 32)
        self.residual_block3 = layers[6] #(, 256, 8, 16)
        self.residual_block4 = layers[7] #(, 512, 4, 8)

        del resnet
        del layers

        #1296 (audio-visual feature) = 784 (visual feature) + 512 (audio feature)
        # self.audionet_upconvlayer1 = unet_upconv(1296, ngf * 8) 

        # channel number
        #1024 = 512 (visual feature) + 512 (audio feature)
        self.audionet_upconvlayer1 = unet_upconv(1024, ngf*8) 
        self.audionet_upconvlayer2 = unet_upconv(ngf*24, ngf*4)
        self.audionet_upconvlayer3 = unet_upconv(ngf*12, ngf*2)
        self.audionet_upconvlayer4 = unet_upconv(ngf*6, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf*3, output_nc, True)
        
        #Visual reshape
        # self.visual_pooling = nn.AdaptiveAvgPool2d((8,2))
        self.visual_conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(2,1), stride=(2,1), padding=0)

    def get_audio_layers(self):
        return [
            self.audionet_convlayer1,
            self.audionet_convlayer2,
            self.audionet_convlayer3,
            self.audionet_convlayer4,
            self.audionet_convlayer5,
            self.audionet_upconvlayer1,
            self.audionet_upconvlayer2,
            self.audionet_upconvlayer3,
            self.audionet_upconvlayer4,
            self.audionet_upconvlayer5,
        ]

    def get_visual_layers(self):
        return [
            self.residual_block1,
            self.residual_block2,
            self.residual_block3,
            self.residual_block4,
        ]

    def forward(self, audio_spec, visual_frame):
        #audio spec (, 2, 257, 64)
        #frame (, 3, 128, 256)

        # Audio encode
        audio_conv1feature = self.audionet_convlayer1(audio_spec)#(, 64, 128, 32)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)#(, 128, 64, 16)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)#(, 256, 32, 8)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)#(, 512, 16, 4)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)# (, 512, 8, 2)

        # Video encode
        video_res1feature = self.residual_block1(visual_frame) #(, 64, 32, 64)
        video_res2feature = self.residual_block2(video_res1feature) #(, 128, 16, 32)
        video_res3feature = self.residual_block3(video_res2feature) #(, 256, 8, 16)
        video_res4feature = self.residual_block4(video_res3feature) #(, 512, 4, 8)
            
        #Conv
        video_res5feature = self.visual_conv(video_res4feature)
        video_res5feature = video_res5feature.transpose(2,3)
        audioVisual_feature = torch.cat((video_res5feature, audio_conv5feature), dim=1)
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)

        # Skip connection
        video_res4feature = video_res4feature.transpose(2,3) 
        video_res4feature = video_res4feature.repeat(1,1,2,1)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((
            audio_upconv1feature, 
            audio_conv4feature,
            video_res4feature), dim=1))

        video_res3feature = video_res3feature.transpose(2,3) 
        video_res3feature = video_res3feature.repeat(1,1,2,1)
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((
            audio_upconv2feature, 
            audio_conv3feature,
            video_res3feature), dim=1))

        video_res2feature = video_res2feature.transpose(2,3) 
        video_res2feature = video_res2feature.repeat(1,1,2,1)
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((
            audio_upconv3feature, 
            audio_conv2feature,
            video_res2feature), dim=1))

        # sigmoid output [0,1], map to [-1,1] 
        video_res1feature = video_res1feature.transpose(2,3) 
        video_res1feature = video_res1feature.repeat(1,1,2,1)
        mask_prediction = self.audionet_upconvlayer5(torch.cat((
            audio_upconv4feature, 
            audio_conv1feature,
            video_res1feature), dim=1)) * 2 - 1

        return mask_prediction
