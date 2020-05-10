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
        self.visual_global = networks.VisualNetGlobal()
        self.visual_cropped = networks.VisualNetCropped()

        self.audio_gen.apply(networks.weights_init)
        self.visual_global.apply(networks.weights_init)
        self.visual_cropped.apply(networks.weights_init)

    def forward(self, input, volatile=False):
        # visual_feature = input['visual_feature'].cuda()
        frame = input['frame'].cuda()
        frame_cropped = input['frame_cropped'].cuda()
        mix_mag = input['mix_mag'].cuda()

        visual_feature_globals = self.visual_global(frame)
        visual_feature_cropped = self.visual_cropped(frame_cropped)
        mask_prediction = self.audio_gen(mix_mag, visual_feature_globals, visual_feature_cropped)

        return mask_prediction
