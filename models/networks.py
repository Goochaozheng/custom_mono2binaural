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

def create_upconv(input_channels, output_channels, kernel=4, paddings=1, stride=2, batch_norm=True, dropout=False, Relu=True):
    model = [nn.ConvTranspose2d(input_nc, output_nc, kernel_size=kernel, stride=2, padding=1)]
    if(dropout):
        model.append(nn.Dropout2d(0.3, True))
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))

    if(Relu):
        model.append(nn.LeakyReLU(0.2, True))
    else:
        model.append(nn.Sigmoid())
    return nn.Sequential(*model)
        

def create_conv(input_channels, output_channels, kernel=4, paddings=1, stride=2, batch_norm=True, dropout=False, Relu=True):
    model = [nn.Conv2d(input_channels, output_channels, kernel_size=kernel, stride=stride, padding=paddings)]
    if dropout:
        model.append(nn.Dropout2d(0.3, True))
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



class FlowNet(nn.Module):
    def __init__(self):
        super(FlowNet, self).__init__()
        self.flow_conv1 = create_conv(3, 32)
        self.flow_conv2 = create_conv(16, 64)
        self.flow_conv3 = create_conv(64, 128)
        self.flow_conv4 = create_conv(128, 256)
        self.flow_conv5 = create_conv(256, 512)

    def forward(self, flow):
        flow_feature = self.flow_conv1(flow)
        flow_feature = self.flow_conv2(flow_feature)
        flow_feature = self.flow_conv3(flow_feature)
        flow_feature = self.flow_conv4(flow_feature)
        flow_feature = self.flow_conv5(flow_feature)
        return flow_feature


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
        self.audionet_convlayer1 = create_conv(input_nc, ngf)
        self.audionet_convlayer2 = create_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = create_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = create_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = create_conv(ngf * 8, ngf * 8)

        self.audionet_upconvlayer1 = create_upconv(1024, ngf * 8) #1296 (audio-visual feature) = 256 (visual feature) + 256 (flow_feature) + 512 (audio feature)
        self.audionet_upconvlayer2 = create_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer3 = create_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = create_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = create_upconv(ngf * 2, output_nc, Relu=False) #outermost layer use a sigmoid to bound the mask
        
        self.conv1x1_visual = create_conv(512, 8, kernel=1, stride=1, paddings=0) 
        self.conv1x1_flow = create_conv(512, 8, kernel=1, stride=1, paddings=0)

    def forward(self, audio, visual_feat, flow_feat):
        audio_conv1feature = self.audionet_convlayer1(audio)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        visual_feat = self.conv1x1_visual(visual_feat)
        visual_feat = visual_feat.view(visual_feat.shape[0], -1, 1, 1) #flatten visual feature
        visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1]) #tile visual feature
        
        flow_feat = self.conv1x1_flow(flow_feat)
        flow_feat = flow_feat.view(flow_feat.shape[0], -1, 1, 1)
        flow_feat = flow_feat.repeat(1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1]) #tile visual feature

        audioVisual_feature = torch.cat((visual_feat, flow_feat, audio_conv5feature), dim=1)
        
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        mask_prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1)) * 2 - 1

        return mask_prediction
