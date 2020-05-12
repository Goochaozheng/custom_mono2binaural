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
    mag = np.abs(spectro)
    phase =  np.angle(spectro)
    # mag_phase = np.concatenate((mag, phase), axis=0)
    return mag, phase

def audio_normalize(samples, desired_rms = 0.1, eps = 1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
  samples = samples * (desired_rms / rms)
  return rms / desired_rms, samples

def frame_normalize(frame):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return normalize(frame)

def inverse_spectrogram(mag, phase):
    spec = mag.astype(np.complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=160, win_length=400, center=True, length=samples_per_window)
    return wav

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

            #########################
            # Process audio data
            #########################
            sliding_window_end = sliding_window_start + samples_per_window
            normalizer, audio_segment = audio_normalize(audio[:,sliding_window_start:sliding_window_end])
            audio_segment_channel_left = audio_segment[0,:]
            audio_segment_channel_right = audio_segment[1,:]
            audio_segment_mix = audio_segment_channel_left + audio_segment_channel_right

            audio_mix_mag,  audio_mix_phase = generate_spectrogram(audio_segment_mix)
            audio_mix_mag = audio_mix_mag[:-1, :]
            audio_mix_phase = audio_mix_phase[:-1, :]
            audio_mix_mag = torch.FloatTensor(audio_mix_mag).unsqueeze(0) # add channel dim
            data['mix_mag'] = torch.log(audio_mix_mag).unsqueeze(0).cuda() # add batch dim


            #########################
            # Process visual data
            #########################
            frame_index = int(((sliding_window_start + samples_per_window / 2.0) / audio.shape[-1]) * frame_count)
            if frame_index > frame_count: frame_index = frame_count

            #Read frame
            frame = Image.open(os.path.join(frame_path, str(frame_index).zfill(6) + '.png'))
            frame = frame.resize((256,128))

            w, h = frame.size
            frame_left = frame.crop((0,0,w/2,h))
            frame_right = frame.crop((w/2,0,w,h))

            frame = transforms.ToTensor()(frame)
            frame = frame_normalize(frame)
            data['frame'] = frame.unsqueeze(0).cuda()


            #########################
            # Generate prediction
            #########################
            frame_left = frame_normalize(frame_left)
            data['frame_cropped'] = transforms.ToTensor()(frame_left).unsqueeze(0).cuda()
            data['audio_cropped'] = torch.FloatTensor(generate_spectrogram(audio_segment_channel_left)).unsqueeze(0).cuda()
            with torch.no_grad():
                output = model.forward(data)
            predicted_mask_left = output[0,:,:,:].data[:].cpu().numpy()

            # generate right cahnnel
            frame_right = frame_normalize(frame_right)
            data['frame_cropped'] = transforms.ToTensor()(frame_right).unsqueeze(0).cuda()
            data['audio_cropped'] = torch.FloatTensor(generate_spectrogram(audio_segment_channel_right)).unsqueeze(0).cuda()
            with torch.no_grad():
                output = model.forward(data)
            predicted_mask_right = output[0,:,:,:].data[:].cpu().numpy()


            #########################
            # Mask to wav
            #########################
            pred_mag_left = audio_mix_mag[0] * predicted_mask_left[0]
            reconstructed_signal_left = inverse_spectrogram(pred_mag_left, audio_mix_phase)

            pred_mag_right = audio_mix_mag[0] * predicted_mask_right[0]
            reconstructed_signal_right = inverse_spectrogram(pred_mag_right, audio_mix_phase)

            reconstructed_binaural = np.concatenate((np.expand_dims(reconstructed_signal_left, axis=0), np.expand_dims(reconstructed_signal_right, axis=0)), axis=0) 
            # inverse normalization
            reconstructed_binaural = reconstructed_binaural * normalizer

            binaural_audio[:,sliding_window_start:sliding_window_end] = binaural_audio[:,sliding_window_start:sliding_window_end] + reconstructed_binaural
            overlap_count[:,sliding_window_start:sliding_window_end] = overlap_count[:,sliding_window_start:sliding_window_end] + 1
            #move to next window
            sliding_window_start = sliding_window_start + int(opt.hop_size * opt.audio_sampling_rate)


        #deal with the last segment
        normalizer, audio_segment = audio_normalize(audio[:,-samples_per_window:])
        audio_segment_channel_left = audio_segment[0,:]
        audio_segment_channel_right = audio_segment[1,:]
        audio_segment_mix = audio_segment_channel_left + audio_segment_channel_right

        audio_mix_mag,  audio_mix_phase = generate_spectrogram(audio_segment_mix)
        audio_mix_mag = audio_mix_mag[:-1, :]
        audio_mix_phase = audio_mix_phase[:-1, :]
        audio_mix_mag = torch.FloatTensor(audio_mix_mag).unsqueeze(0) # add channel dim
        data['mix_mag'] = torch.log(audio_mix_mag).unsqueeze(0).cuda() # add batch dim
        
        frame_index = int(round((opt.input_audio_length - opt.audio_length / 2.0) * 10))
        if frame_index > frame_count: frame_index = frame_count
        frame = Image.open(os.path.join(frame_path, str(frame_index).zfill(6) + '.png'))

        #check output directory
        if not os.path.isdir(os.path.join(opt.output_dir_root, audio_name)):
            os.mkdir(os.path.join(opt.output_dir_root, audio_name))
        #save sample image
        frame.save(os.path.join(opt.output_dir_root, audio_name, 'sample_image.png'))

        frame = frame.resize((256,128))
        w, h = frame.size
        frame_left = frame.crop((0,0,w/2,h))
        frame_right = frame.crop((w/2,0,w,h))

        frame = transforms.ToTensor()(frame)
        frame = frame_normalize(frame)
        data['frame'] = frame.unsqueeze(0).cuda()

        frame_left = frame_normalize(frame_left)
        data['frame_cropped'] = transforms.ToTensor()(frame_left).unsqueeze(0).cuda()
        data['audio_cropped'] = torch.FloatTensor(generate_spectrogram(audio_segment_channel_left)).unsqueeze(0).cuda()
        with torch.no_grad():
            output = model.forward(data)
        predicted_mask_left = output[0,:,:,:].data[:].cpu().numpy()

        frame_right = frame_normalize(frame_right)
        data['frame_cropped'] = transforms.ToTensor()(frame_right).unsqueeze(0).cuda()
        data['audio_cropped'] = torch.FloatTensor(generate_spectrogram(audio_segment_channel_right)).unsqueeze(0).cuda()
        with torch.no_grad():
            output = model.forward(data)
        predicted_mask_right = output[0,:,:,:].data[:].cpu().numpy()

        pred_mag_left = audio_mix_mag[0] * predicted_mask_left[0]
        reconstructed_signal_left = inverse_spectrogram(pred_mag_left, audio_mix_phase)

        pred_mag_right = audio_mix_mag[0] * predicted_mask_right[0]
        reconstructed_signal_right = inverse_spectrogram(pred_mag_right, audio_mix_phase)

        reconstructed_binaural = np.concatenate((np.expand_dims(reconstructed_signal_left, axis=0), np.expand_dims(reconstructed_signal_right, axis=0)), axis=0) * normalizer

        #add the spatialized audio to reconstructed_binaural
        binaural_audio[:,-samples_per_window:] = binaural_audio[:,-samples_per_window:] + reconstructed_binaural
        overlap_count[:,-samples_per_window:] = overlap_count[:,-samples_per_window:] + 1

        #divide aggregated predicted audio by their corresponding counts
        predicted_binaural_audio = np.divide(binaural_audio, overlap_count)

        mixed_mono = (audio_channel1 + audio_channel2) / 2

        librosa.output.write_wav(os.path.join(opt.output_dir_root, audio_name, 'mixed_mono.wav'), mixed_mono, sr=opt.audio_sampling_rate)
        librosa.output.write_wav(os.path.join(opt.output_dir_root, audio_name, 'input_binaural.wav'), audio, sr=opt.audio_sampling_rate)
        librosa.output.write_wav(os.path.join(opt.output_dir_root, audio_name, 'predicted_binaural.wav'), predicted_binaural_audio, sr=opt.audio_sampling_rate)


if __name__ == '__main__':
    main()
