import cv2
import librosa
import numpy as np
import os
import random
from tqdm import tqdm
import torch
import torchvision

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

    audio_dir = "H:\\FAIR-Play\\FAIR-Play\\binaural_audios"
    video_dir = "H:\\FAIR-Play\\FAIR-Play\\videos"
    audio_files = os.listdir(audio_dir)
    audio_files = [os.path.join(audio_dir, i) for i in audio_files]
    video_files = os.listdir(video_dir)
    video_files = [os.path.join(video_dir, i) for i in video_files]

    segment_length = 0.63
    output_root = 'data\\split-2\\'

    audio_mix_spec_list = []
    audio_diff_spec_list = []

    # Load pre-trained resnet18
    resnet = torchvision.models.resnet18(pretrained=True)
    layers = list(resnet.children())[0:-2]
    visual_extraction = torch.nn.Sequential(*layers) 

    for i in tqdm(range(len(audio_files))):
        
        # Load audio
        audio, sr = librosa.load(audio_files[i], mono=False, sr=16000)
        audio_len = librosa.get_duration(audio)

        # Slice audio
        # Length of the binaural audio and video is weakly consistent, limit the segmentation between [0, 9.90]
        start_time = round(random.uniform(0, 9.9 - segment_length), 2)
        end_time = start_time + segment_length
        start_pos = int(start_time*sr)
        end_pos = start_pos + int(segment_length * sr)
        audio_seg = audio[:, start_pos : end_pos]

        audio_seg = normalize(audio_seg)
        audio_channel_1 = audio_seg[0,:]
        audio_channel_2 = audio_seg[1,:]
        audio_mix = audio_channel_1 + audio_channel_2
        audio_diff = audio_channel_1 - audio_channel_2
        
        #audio spectrogram
        audio_mix_spec = generate_spectrogram(audio_mix)
        audio_diff_spec = generate_spectrogram(audio_diff)

        audio_mix_spec_list.append(audio_mix_spec)
        audio_diff_spec_list.append(audio_diff_spec)


        # Load video    
        video = cv2.VideoCapture(video_files[i])
        fps = video.get(cv2.CAP_PROP_FPS)
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        size = (width, height)
        fourcc = int(video.get(cv2.CAP_PROP_FOURCC))

        # Capture center frame
        center_frame_pos = int(round(((start_time + end_time) / 2) * fps))
        video.set(cv2.CAP_PROP_POS_FRAMES, center_frame_pos)
        flag, frame = video.read()
        if not flag:
            print("frame fail: %d" % center_frame_pos)
            print("total frame: %d" % video.get(cv2.CAP_PROP_FRAME_COUNT))
            print("seg: %f - %f" % (start_time, end_time))
            exit()

        # extract visual feature
		frame = Image.fromarray(frame)
		frame = vision_transform(frame).unsqueeze(0)
        visual_feature = visual_extraction(frame)

        # # Slice video
        # start_frame = int(start_time * fps)
        # end_frame = int(end_time * fps)
        # # fourcc = cv2.VideoWriter_fourcc(*'H264')
        # out_seg = cv2.VideoWriter(output_root + "video_seg\\%06d.mp4" % (i+1), fourcc, fps, size)
        # video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # for f in range(start_frame, end_frame+1):
        #     success, frame = video.read()
        #     if success:
        #         out_seg.write(frame)
        #     else:
        #         print("read fail: %d\n" % f)
        #         print("total frame: %d" % video.get(cv2.CAP_PROP_FRAME_COUNT))
        #         exit()
        # video.release()
        # out_seg.release()

