#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import librosa
import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import subprocess
from options.test_options import TestOptions
import torchvision
import torchvision.transforms as transforms
import torch
from models.audioVisual_model import AudioVisualModel
from dataloader.custom_dataset import CustomDataset

def generate_spectrogram(audio):
    spectro = librosa.core.stft(audio, n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel


def audio_normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return rms / desired_rms, samples


def main():
    #load test arguments
    opt = TestOptions().parse()
    device = torch.device("cuda:0")
    opt.mode = 'test'

    # load model
    model = torch.load(opt.model_path)
    model.to(device)
    model.eval()

    loss_criterion = torch.nn.MSELoss()

    data_source = h5py.File(opt.data_path)
    audio_source = h5py.File(opt.input_audio_path)

    for index in range(len(data_source['audio'])):

        path_parts = data_source['audio'][index].decode().strip().split('\\')
        audio_name = path_parts[-1][:-4]
        audio_index = int(audio_name) - 1

        print("Processing audio: %s (%d out of %d)" % (audio_name, index, len(data_source['audio'])))

        #load the audio to perform separation
        audio = audio_source['audio'][audio_index]
        audio_channel1 = audio[0,:]
        audio_channel2 = audio[1,:]

        #load video 
        frame_path = "H:\\FAIR-Play\\FAIR-Play\\frames\\" + audio_name
        frame_count = len(os.listdir(frame_path))

        #perform spatialization over the whole audio using a sliding window approach
        overlap_count = np.zeros((audio.shape)) #count the number of times a data point is calculated
        binaural_audio = np.zeros((audio.shape))

        #perform spatialization over the whole spectrogram in a siliding-window fashion
        sliding_window_start = 0
        data = {}
        samples_per_window = int(opt.audio_length * opt.audio_sampling_rate)
        total_loss = 0
        count = 0

        while sliding_window_start + samples_per_window < audio.shape[-1]:

            sliding_window_end = sliding_window_start + samples_per_window
            normalizer, audio_segment = audio_normalize(audio[:,sliding_window_start:sliding_window_end])
            audio_segment_channel_left = audio_segment[0,:]
            audio_segment_channel_right = audio_segment[1,:]
            audio_segment_mix = audio_segment_channel1 + audio_segment_channel2

            data['audio_mix'] = torch.FloatTensor(generate_spectrogram(audio_segment_channel_left + audio_segment_channel_right)).unsqueeze(0)
            #get the frame index for current window
            frame_index = int((((sliding_window_start + samples_per_window / 2.0) / audio.shape[-1]) * opt.input_audio_length) * 10)

            #Read frame
            frame = Image.open(os.path.join(frame_path, str(frame_index).zfill(6) + '.png'))
            frame = frame.resize((246,128))
			data['frame'] = transforms.ToTensor()(frame).unsqueeze(0).cuda()

            w, h = frame.size
            frame_left = frame.crop((0,0,w/2,h))
            frame_right = frame.crop((w/2,0,w,h))

            # generate left cahnnel
            data['frame_cropped'] = transforms.ToTensor()(frame_left).unsqueeze(0).cuda()
            data['audio_cropped'] = torch.FloatTensor(generate_spectrogram(audio_segment_channel_left)).unsqueeze(0).cuda()
            with torch.no_grad():
                output = model.forward(data)
			predicted_spectrogram_left = output[0,:,:,:].data[:].cpu().numpy()
            loss = loss_criterion(output, data['audio_cropped'][:,:,:-1,:].cuda())
            total_loss = total_loss + loss
            count = count + 1

            # generate right cahnnel
            data['frame_cropped'] = transforms.ToTensor()(frame_right).unsqueeze(0).cuda()
            data['audio_cropped'] = torch.FloatTensor(generate_spectrogram(audio_segment_channel_right)).unsqueeze(0).cuda()
            with torch.no_grad():
                output = model.forward(data)
			predicted_spectrogram_right = output[0,:,:,:].data[:].cpu().numpy()
            loss = loss_criterion(output, data['audio_cropped'][:,:,:-1,:].cuda())
            total_loss = total_loss + loss
            count = count + 1


            #ISTFT to convert back to audio
            reconstructed_stft_left = predicted_spectrogram_left[0,:,:] + (1j * predicted_spectrogram_left[1,:,:])
            reconstructed_signal_left = librosa.istft(reconstructed_stft_left, hop_length=160, win_length=400, center=True, length=samples_per_window)

            reconstructed_stft_right = predicted_spectrogram_right[0,:,:] + (1j * predicted_spectrogram_right[1,:,:])
            reconstructed_signal_right = librosa.istft(reconstructed_stft_right, hop_length=160, win_length=400, center=True, length=samples_per_window)

            reconstructed_binaural = np.concatenate((np.expand_dims(reconstructed_signal_left, axis=0), np.expand_dims(reconstructed_signal_right, axis=0)), axis=0) 
            # inverse normalization
            reconstructed_binaural = reconstructed_binaural * normalizer

            binaural_audio[:,sliding_window_start:sliding_window_end] = binaural_audio[:,sliding_window_start:sliding_window_end] + reconstructed_binaural
            overlap_count[:,sliding_window_start:sliding_window_end] = overlap_count[:,sliding_window_start:sliding_window_end] + 1
            #move to next window
            sliding_window_start = sliding_window_start + int(opt.hop_size * opt.audio_sampling_rate)

        #deal with the last segment
        normalizer, audio_segment = audio_normalize(audio[:,-samples_per_window:])
        audio_segment_channel1 = audio_segment[0,:]
        audio_segment_channel2 = audio_segment[1,:]
        data['audio_diff'] = torch.FloatTensor(generate_spectrogram(audio_segment_channel1 - audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
        data['audio_mix'] = torch.FloatTensor(generate_spectrogram(audio_segment_channel1 + audio_segment_channel2)).unsqueeze(0) #unsqueeze to add a batch dimension
        #get the frame index for last window
        
        frame_index = int(round((opt.input_audio_length - opt.audio_length / 2.0) * 10))
        if frame_index > frame_count: frame_index = frame_count
        frame = Image.open(os.path.join(frame_path, str(frame_index).zfill(6) + '.png'))

        #check output directory
        if not os.path.isdir(os.path.join(opt.output_dir_root, audio_name)):
            os.mkdir(os.path.join(opt.output_dir_root, audio_name))
        #save sample image
        frame.save(os.path.join(opt.output_dir_root, audio_name, 'sample_image.png'))
        frame = frame.resize((246,128))
        data['frame'] = transforms.ToTensor()(frame).unsqueeze(0).cuda()

        w, h = frame.size
        frame_left = frame.crop((0,0,w/2,h))
        frame_right = frame.crop((w/2,0,w,h))

        # generate left cahnnel
        data['frame_cropped'] = transforms.ToTensor()(frame_left).unsqueeze(0).cuda()
        data['audio_cropped'] = torch.FloatTensor(generate_spectrogram(audio_segment_channel_left)).unsqueeze(0).cuda()
        with torch.no_grad():
            output = model.forward(data)
        predicted_spectrogram_left = output[0,:,:,:].data[:].cpu().numpy()
        loss = loss_criterion(output, data['audio_cropped'][:,:,:-1,:].cuda())
        total_loss = total_loss + loss
        count = count + 1

        # generate right cahnnel
        data['frame_cropped'] = transforms.ToTensor()(frame_right).unsqueeze(0).cuda()
        data['audio_cropped'] = torch.FloatTensor(generate_spectrogram(audio_segment_channel_right)).unsqueeze(0).cuda()
        with torch.no_grad():
            output = model.forward(data)
        predicted_spectrogram_right = output[0,:,:,:].data[:].cpu().numpy()
        loss = loss_criterion(output, data['audio_cropped'][:,:,:-1,:].cuda())
        total_loss = total_loss + loss
        count = count + 1

        #ISTFT to convert back to audio
        reconstructed_stft_left = predicted_spectrogram_left[0,:,:] + (1j * predicted_spectrogram_left[1,:,:])
        reconstructed_signal_left = librosa.istft(reconstructed_stft_left, hop_length=160, win_length=400, center=True, length=samples_per_window)

        reconstructed_stft_right = predicted_spectrogram_right[0,:,:] + (1j * predicted_spectrogram_right[1,:,:])
        reconstructed_signal_right = librosa.istft(reconstructed_stft_right, hop_length=160, win_length=400, center=True, length=samples_per_window)

        reconstructed_binaural = np.concatenate((np.expand_dims(reconstructed_signal_left, axis=0), np.expand_dims(reconstructed_signal_right, axis=0)), axis=0) * normalizer

        #add the spatialized audio to reconstructed_binaural
        binaural_audio[:,-samples_per_window:] = binaural_audio[:,-samples_per_window:] + reconstructed_binaural
        overlap_count[:,-samples_per_window:] = overlap_count[:,-samples_per_window:] + 1

        #divide aggregated predicted audio by their corresponding counts
        predicted_binaural_audio = np.divide(binaural_audio, overlap_count)

        print('Loss:%f' % (total_loss/count))

        mixed_mono = (audio_channel1 + audio_channel2) / 2

        librosa.output.write_wav(os.path.join(opt.output_dir_root, audio_name, 'mixed_mono.wav'), mixed_mono, sr=opt.audio_sampling_rate)
        librosa.output.write_wav(os.path.join(opt.output_dir_root, audio_name, 'input_binaural.wav'), audio, sr=opt.audio_sampling_rate)
        librosa.output.write_wav(os.path.join(opt.output_dir_root, audio_name, 'predicted_binaural.wav'), predicted_binaural_audio, sr=opt.audio_sampling_rate)


if __name__ == '__main__':
    main()
