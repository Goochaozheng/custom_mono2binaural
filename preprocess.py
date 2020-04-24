import cv2
import librosa
import numpy as np
import os
import random
from tqdm import tqdm
import torch
import torchvision
import h5py

AUDIO_DIR = "H:\\FAIR-Play\\FAIR-Play\\binaural_audios"
VIDEO_DIR = "H:\\FAIR-Play\\FAIR-Play\\videos"

SEGMENT_LENGTH = 0.63
DATASET_SIZE = 5000
OUTPUT_DIR = 'data\\split-6\\'

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


def frame_tranform(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (256,128))
    # normalize = torchvision.transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std=[0.229, 0.224, 0.225]
    # )
    # vision_transform_list = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), normalize])
    frame = torchvision.transforms.ToTensor()(frame).numpy()
    return frame


def write_hdf5(file_name, data):
    data_name = file_name[:-3]
    with h5py.File(OUTPUT_DIR+file_name, 'a') as f:
        # Create dataest
        if len(f.keys()) == 0:
            data_maxshape = tuple([None]) + data[0].shape
            f.create_dataset(name=data_name, chunks=True, maxshape=data_maxshape, data=data)
        else: #insert data
            dataset = f[data_name]
            data_size = len(dataset)
            dataset.resize(len(dataset)+len(data), axis=0)
            dataset[-len(data):] = data


def main():

    audio_files = os.listdir(AUDIO_DIR)

    audio_mix_spec_list = np.empty(shape=(0,2,257,64))
    audio_diff_spec_list = np.empty(shape=(0,2,257,64))
    # visual_feature_list = np.empty(shape=(0,512,4,8))
    frame_list = np.empty(shape=(0,3,128,256))

    # Load pre-trained resnet18
    # resnet = torchvision.models.resnet18(pretrained=True)
    # layers = list(resnet.children())[0:-2]
    # visual_extraction = torch.nn.Sequential(*layers) 
    # visual_extraction.to("cuda:0")

    for i in tqdm(range(DATASET_SIZE), ascii=True):
        
        audio_file = random.choice(audio_files)
        audio_path = os.path.join(AUDIO_DIR, audio_file)

        # Load audio
        audio, sr = librosa.load(audio_path, mono=False, sr=16000)
        # audio_len = librosa.get_duration(audio)

        # Slice audio
        # Length of the binaural audio and video is weakly consistent, limit the segmentation between [0, 8.90]
        start_time = round(random.uniform(0, 8.9 - SEGMENT_LENGTH), 2)
        end_time = start_time + SEGMENT_LENGTH
        start_pos = int(start_time*sr)
        end_pos = start_pos + int(SEGMENT_LENGTH * sr)
        audio_seg = audio[:, start_pos : end_pos]

        audio_seg = audio_normalize(audio_seg)
        audio_channel_1 = audio_seg[0,:]
        audio_channel_2 = audio_seg[1,:]
        audio_mix = audio_channel_1 + audio_channel_2
        audio_diff = audio_channel_1 - audio_channel_2
        
        #audio spectrogram
        audio_mix_spec = generate_spectrogram(audio_mix)
        audio_diff_spec = generate_spectrogram(audio_diff)

        audio_mix_spec_list = np.concatenate([audio_mix_spec_list, [audio_mix_spec]])
        audio_diff_spec_list = np.concatenate([audio_diff_spec_list, [audio_diff_spec]])

        # write to hdf5
        if (i+1)%100 == 0 or i == DATASET_SIZE-1:
            #write audio mix
            write_hdf5('audio_mix_spec.h5', audio_mix_spec_list)
            audio_mix_spec_list = np.empty(shape=(0,2,257,64))

            #write audio diff
            write_hdf5('audio_diff_spec.h5', audio_diff_spec_list)
            audio_diff_spec_list = np.empty(shape=(0,2,257,64))

        #clear audio var
        del audio

        #############################################
        # Load video    
        #############################################
        video_file = audio_file[:-4] + '.mp4'
        video_path = os.path.join(VIDEO_DIR, video_file)
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)

        # Capture center frame
        center_frame_pos = int(round(((start_time + end_time) / 2) * fps))
        video.set(cv2.CAP_PROP_POS_FRAMES, center_frame_pos)
        flag, frame = video.read()
        if not flag:
            print("frame fail: %d" % center_frame_pos)
            print("total frame: %d" % video.get(cv2.CAP_PROP_FRAME_COUNT))
            print("seg: %f - %f" % (start_time, end_time))
            exit()

        #frame process
        frame = frame_tranform(frame)

        # extract visual feature
        # with torch.no_grad():
        #     visual_feature = visual_extraction(frame)

        # visual_feature = visual_feature.cpu().numpy()
        # visual_feature_list = np.concatenate([visual_feature_list, visual_feature])

        frame_list = np.concatenate([frame_list, [frame]])

        # write to hdf5
        if (i+1)%100 == 0 or i == DATASET_SIZE-1:
            #write visual feature
            write_hdf5('frame.h5', frame_list)
            frame_list = np.empty(shape=(0,3,128,256))
      
        del video      

if __name__ == '__main__':
    main()

