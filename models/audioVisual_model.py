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
        self.attention_net = networks.AttentionNet()
        self.visual_extraction = networks.VisualNet()
        self.audio_extraction = networks.AudioNet()
        self.audio_generation = networks.GenerationNet()
    
    def forward(self, input, volatile=False):
        frame = input['frame'].cuda()
        audio_mix = input['audio_mix'].cuda()

        attention_mask = self.attention_net(frame)
        frame = attention_mask * frame

        visual_feature = self.visual_extraction(frame)
        audio_feature = self.audio_extraction(audio_mix)

        visual_feature = torch.transpose(visual_feature, 2, 3)#(, 512, 8, 2) -> (, 512, 2, 8)
        audio_visual_feature = torch.cat((audio_feature, visual_feature), dim=1)

        mask_prediction = self.audio_generation(audio_visual_feature)

        # mask_prediction (, 2, 256, 64)
        spectrogram_diff_real = audio_mix[:,0,:-1,:] * mask_prediction[:,0,:,:] - audio_mix[:,1,:-1,:] * mask_prediction[:,1,:,:]
        spectrogram_diff_img = audio_mix[:,0,:-1,:] * mask_prediction[:,1,:,:] + audio_mix[:,1,:-1,:] * mask_prediction[:,0,:,:]
        
        binaural_spectrogram = torch.cat((spectrogram_diff_real.unsqueeze(1), spectrogram_diff_img.unsqueeze(1)), 1)

        return {
            "attention_mask": attention_mask,
            "binaural_spec": binaural_spectrogram
        }
