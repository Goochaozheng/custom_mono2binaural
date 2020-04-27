import torch
import h5py
import librosa
import matplotlib.pyplot as plt
import numpy as np
import os
import torchvision.transforms as transforms
import random

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

    def __len__(self):
        return len(self.data_source['audio'])

    def __getitem__(self, index):

        audio, audio_rate = librosa.load(self.data_source['audio'][index].decode(), sr=self.opt.audio_sampling_rate, mono=False)

        #randomly get a start time for the audio segment from the 10s clip
        audio_start_time = random.uniform(0, 9 - self.opt.audio_length)
        audio_end_time = audio_start_time + self.opt.audio_length
        audio_start = int(audio_start_time * self.opt.audio_sampling_rate)
        audio_end = audio_start + int(self.opt.audio_length * self.opt.audio_sampling_rate)
        audio = audio[:, audio_start:audio_end]
        audio = normalize(audio)
        audio_channel1 = audio[0,:]
        audio_channel2 = audio[1,:]

        #passing the spectrogram of the difference
        audio_diff_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 - audio_channel2))
        audio_mix_spec = torch.FloatTensor(generate_spectrogram(audio_channel1 + audio_channel2))

        #get the frame dir path based on audio path
        path_parts = self.audios[index].decode().strip().split('\\')
        path_parts[-1] = path_parts[-1][:-4]
        path_parts[-2] = 'frames'
        frame_path = '/'.join(path_parts)

        frame_index = int(round(((audio_start_time + audio_end_time) / 2.0 + 0.05) * 10))  #10 frames extracted per second
        frame = Image.open(os.path.join(frame_path, str(frame_index).zfill(6) + '.png'))
        frame = frame.resize((256,128))
        frame = transforms.ToTensor()(frame)

        data = {
            # 'visual_feature': visual_feature,
            'frame': frame,
            'audio_mix': audio_mix_spec,
            'audio_diff': audio_diff_spec
        }
        return data

    