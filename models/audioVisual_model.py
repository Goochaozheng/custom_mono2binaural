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

        self.u_net.apply(networks.weights_init)
        
        self.visual_extract = networks.VisualNet()

    def forward(self, input, volatile=False):
        frame = input['frame'].cuda()
        audio_mix = input['audio_mix'].cuda()

        visual_feature = self.visual_extract(frame)
        predicted_diff = self.u_net(audio_mix, visual_feature) 
        
        return predicted_diff
