#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Define the whole model of generating 2.5D visual sound

import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from . import networks,criterion
from torch.autograd import Variable
import torchvision

class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt
        #initialize model
        self.u_net = networks.AudioNet(
            ngf=opt.unet_ngf,
            input_nc=opt.unet_input_nc,
            output_nc=opt.unet_output_nc
        )
        self.visual_extract = networks.VisualNet()
        self.visual_fusion = networks.VisualFusion()

        # self.visual_extract.apply(networks.weights_init)
        self.visual_fusion.apply(networks.weights_init)
        self.u_net.apply(networks.weights_init)
        
    def forward(self, input, volatile=False):
        audio_mix = input['audio_mix'].cuda()
        frame_l = input['frame_left'].cuda()
        frame_r = input['frame_right'].cuda()

        visual_feature_left = self.visual_extract(frame_l)
        visual_feature_right = self.visual_extract(frame_r)
        visual_feature = self.visual_fusion(visual_feature_left, visual_feature_right)
        mask_prediction = self.u_net(audio_mix, visual_feature) # U-Net

        spectrogram_diff_real = audio_mix[:,0,:-1,:] * mask_prediction[:,0,:,:] - audio_mix[:,1,:-1,:] * mask_prediction[:,1,:,:]
        spectrogram_diff_img = audio_mix[:,0,:-1,:] * mask_prediction[:,1,:,:] + audio_mix[:,1,:-1,:] * mask_prediction[:,0,:,:]
        
        binaural_spectrogram = torch.cat((spectrogram_diff_real.unsqueeze(1), spectrogram_diff_img.unsqueeze(1)), 1)

        return binaural_spectrogram
