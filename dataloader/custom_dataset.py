import torch
import h5py
import librosa
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision.transforms as transforms

def generate_spectrogram(audio):
    spectro = librosa.core.stft(audio, n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel

def normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return samples

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
        self.audio_source = h5py.File(opt.audio_source)

    def __len__(self):
        return len(self.data_source['audio'])

    def __getitem__(self, index):

        # randomly select left/right part, 0->left, 1->right
        cropped = int(random.uniform(0,2))

        path_parts = self.data_source['audio'][index].decode().strip().split('\\')
        audio_index = int(path_parts[-1][:-4]) - 1
        audio = self.audio_source['audio'][audio_index]

        #randomly get a start time for the audio segment from the 10s clip
        audio_start_time = random.uniform(0, 9 - self.opt.audio_length)
        audio_end_time = audio_start_time + self.opt.audio_length
        audio_start = int(audio_start_time * self.opt.audio_sampling_rate)
        audio_end = audio_start + int(self.opt.audio_length * self.opt.audio_sampling_rate)
        audio = audio[:, audio_start:audio_end]
        audio = normalize(audio)

        #passing the spectrogram of the difference
        audio_mix_spec = torch.FloatTensor(generate_spectrogram(audio[0,:] + audio[1,:]))
        audio_cropped_spec = torch.FloatTensor(generate_spectrogram(audio[cropped,:]))
        
        #get the frame dir path based on audio path
        path_parts[-1] = path_parts[-1][:-4]
        path_parts[-2] = 'frames'
        frame_path = '/'.join(path_parts)
        frame_count = len(os.listdir(frame_path))

        frame_index = int((audio_start_time + audio_end_time) / 2 * 10)
        if frame_index > frame_count:
            frame_index = frame_count
        frame = Image.open(os.path.join(frame_path, str(frame_index).zfill(6) + '.png'))
        frame = frame.resize((256,128))

        w, h = frame.size
        if cropped == 0:
            frame_cropped = frame.crop((0,0,w/2,h))
        else:
            frame_cropped = frame.crop((w/2,0,w,h))

        frame = transforms.ToTensor()(frame)
        frame_cropped = transforms.ToTensor()(frame_cropped)

        data = {
            # 'visual_feature': visual_feature,
            'frame': frame,
            'frame_cropped': frame_cropped,
            'audio_mix': audio_mix_spec,
            'audio_cropped': audio_cropped_spec
        }
        return data

    