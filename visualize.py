import torch
import cv2
import torchvision
import os
import h5py
import numpy as np
import librosa
import time
import random
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

audio_length = 0.63
audio_sampling_rate = 16000

model_path = "checkpoints/audio_dense_1_1/400_model.pth"

data_path = "data/split-8/test.h5"
input_audio_path = "D:/Workspace/FAIR-Play/FAIR-Play/audio_h5/audio.h5"
input_frame_path = "D:/Workspace/FAIR-Play/FAIR-Play/frames/"

output_dir_root = "demo/visualize/"

def generate_spectrogram(audio):
    spectro = librosa.core.stft(audio, n_fft=512, hop_length=160, win_length=400, center=True)
    real = np.expand_dims(np.real(spectro), axis=0)
    imag = np.expand_dims(np.imag(spectro), axis=0)
    spectro_two_channel = np.concatenate((real, imag), axis=0)
    return spectro_two_channel

def audio_normalize(samples, desired_rms = 0.1, eps = 1e-4):
    rms = np.maximum(eps, np.sqrt(np.mean(samples**2)))
    samples = samples * (desired_rms / rms)
    return samples

def mask_out(frame, col, row):
    frame = np.array(frame)
    x = 4*col
    y = 4*row
    frame[y:y+4, x:x+4, :] = frame.mean()
    return frame

def write_loss(loss_map, col, row, loss):
    x = 4*col
    y = 4*row
    loss_map[y:y+4, x:x+4] = loss
    return loss_map

model = torch.load(model_path).to("cuda:0")
model.eval()
loss_criterion = torch.nn.MSELoss()

data_source = h5py.File(data_path)
audio_source = h5py.File(input_audio_path)


for index in tqdm(range(len(data_source['audio'])), ascii=True):

    data = {}
    loss_map = torch.zeros((128,256)).cuda()

    path_parts = data_source['audio'][index].decode().strip().split('\\')
    audio_name = path_parts[-1][:-4]
    audio_index = int(audio_name) - 1
    audio = audio_source['audio'][audio_index]

    audio_start_time = random.uniform(0, 9 - audio_length)
    audio_end_time = audio_start_time + audio_length
    audio_start = int(audio_start_time * audio_sampling_rate)
    audio_end = audio_start + int(audio_length * audio_sampling_rate)
    input_audio = audio[:, audio_start:audio_end]

    input_audio = audio_normalize(input_audio)
    audio_mix = input_audio[0] + input_audio[1]
    audio_diff = input_audio[0] - input_audio[1]
    data['audio_mix'] = torch.FloatTensor(generate_spectrogram(audio_mix)).unsqueeze(0).cuda()
    data['audio_diff'] = torch.FloatTensor(generate_spectrogram(audio_diff)).unsqueeze(0).cuda()

    frame_path = input_frame_path + audio_name
    frame_index = int(round(((audio_start_time + audio_end_time) / 2.0 + 0.05) * 10))
    frame = Image.open(os.path.join(frame_path, str(frame_index).zfill(6) + '.png'))
    input_image = frame.resize((256,128))

    origin_input = torchvision.transforms.ToTensor()(input_image)
    data['frame'] = origin_input.unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(data)
    origin_loss = loss_criterion(out, data['audio_diff'][:,:,:-1,:].cuda())

    for i in range(32*64):
        col = int(i % 64)
        row = int(i / 64)
        mask_image = mask_out(input_image, col, row)
        mask_image = Image.fromarray(mask_image)
        mask_image = torchvision.transforms.ToTensor()(mask_image)
        data['frame'] = mask_image.unsqueeze(0).cuda()
        
        with torch.no_grad():
            out = model(data)

        loss = loss_criterion(out, data['audio_diff'][:,:,:-1,:].cuda())
        loss_diff = loss - origin_loss
        if loss_diff < 0: loss_diff=0
        loss_map = write_loss(loss_map, col, row, loss_diff)

    if not os.path.isdir(os.path.join(output_dir_root, audio_name)):
        os.mkdir(os.path.join(output_dir_root, audio_name))

    loss_map = loss_map.cpu().numpy()
    np.savetxt(os.path.join(output_dir_root, audio_name, "loss_map.txt"), loss_map)

    loss_map = (loss_map - loss_map.min()) / (loss_map.max() - loss_map.min())
    img = Image.fromarray(np.uint8(loss_map * 255))
    img.save(os.path.join(output_dir_root, audio_name, "loss_map.png"))
    frame.resize((256,128)).save(os.path.join(output_dir_root, audio_name, "input.png"))

