import torch
import h5py
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision.transforms as transforms

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, opt):
        super(CustomDataset, self).__init__()
        self.opt = opt
        if opt.mode == 'val':
            path = os.path.join(opt.data_dir, 'val.h5')
        elif opt.mode == 'train':
            path = os.path.join(opt.data_dir, 'train.h5')
        elif opt.mode == 'test':
            path = os.path.join(opt.data_dir, 'test.h5')            
        self.data_source = h5py.File(path)

    def __len__(self):
        return len(self.data_source['frame'])

    def __getitem__(self, idx):

        frame = transforms.ToTensor()(self.data_source['frame'][idx])
        audio_mix = torch.FloatTensor(self.data_source['audio_mix'][idx])
        aduio_diff = torch.FloatTensor(self.data_source['audio_diff'][idx])

        data = {
            'frame': frame,
            'audio_mix': audio_mix,
            'audio_diff': aduio_diff
        }
        return data

    