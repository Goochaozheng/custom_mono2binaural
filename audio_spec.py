import librosa
import h5py
import numpy as np
import os
from tqdm import tqdm
import torch

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

def main():
    audio_dir = 'data\\split-2\\audio_seg'
    audios = os.listdir(audio_dir)
    audios = [os.path.join(audio_dir, i) for i in audios]

    audio_mix_spec_list = []
    audio_diff_spec_list = []

    for index in tqdm(range(len(audios))):
        audio, sample_rate = librosa.load(audios[index], mono=False, sr=16000)

        audio = normalize(audio)

        channel_1 = audio[0,:]
        channel_2 = audio[1,:]
        audio_mix = channel_1 + channel_2
        audio_diff = channel_1 - channel_2

        audio_mix_spec = generate_spectrogram(audio_mix)
        audio_diff_spec = generate_spectrogram(audio_diff)

        if audio_diff_spec.shape != (2, 257, 64) or audio_mix_spec.shape != (2, 257, 64):
            print(index)
            break

        audio_mix_spec_list.append(audio_mix_spec)
        audio_diff_spec_list.append(audio_diff_spec)
    
    audio_mix_spec_list = np.array(audio_mix_spec_list)
    audio_diff_spec_list = np.array(audio_diff_spec_list)

    with h5py.File('data\\split-2\\audio_mix_spec.h5', 'w') as f:
        f.create_dataset('audio_mix_spec', data=audio_mix_spec_list)

    with h5py.File('data\\split-2\\audio_diff_spec.h5', 'w') as f:
        f.create_dataset('audio_diff_spec', data=audio_diff_spec_list)

if __name__ == "__main__":
    main()