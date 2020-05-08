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
        self.audio_gen = networks.AudioNet(
            ngf=opt.unet_ngf,
            input_nc=opt.unet_input_nc,
            output_nc=opt.unet_output_nc
        )
        self.visual_global = networks.VisualNet()
        self.visual_cropped = networks.VisualNet()

        self.audio_gen.apply(networks.weights_init)
        self.visual_global.apply(networks.weights_init)

    def forward(self, input, volatile=False):
        # visual_feature = input['visual_feature'].cuda()
        frame = input['frame'].cuda()
        frame_cropped = input['frame_cropped'].cuda()
        audio_mix = input['audio_mix'].cuda()
        # audio_cropped = input['audio_cropped'].cuda()

        visual_feature_globals = self.visual_global(frame)
        visual_feature_cropped = self.visual_cropped(frame_cropped)
        mask_prediction = self.audio_gen(audio_mix, visual_feature_globals, visual_feature_cropped)

        # complex masking to obtain the predicted spectrogram by complex multiplying (a+bi)(b+ci) = (ac-bd)+(bc+ad)i
        # mask_prediction (, 2, 256, 64)
        pred_spectrogram_real = audio_mix[:,0,:-1,:] * mask_prediction[:,0,:,:] - audio_mix[:,1,:-1,:] * mask_prediction[:,1,:,:]
        pred_spectrogram_img = audio_mix[:,0,:-1,:] * mask_prediction[:,1,:,:] + audio_mix[:,1,:-1,:] * mask_prediction[:,0,:,:]
        
        pred_spectrogram = torch.cat((pred_spectrogram_real.unsqueeze(1), pred_spectrogram_img.unsqueeze(1)), 1)

        return pred_spectrogram
