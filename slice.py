# Randomly select 1s segment from original video

import cv2
import librosa
import numpy as np
import os
import random
from tqdm import tqdm

audio_dir = "H:\\FAIR-Play\\FAIR-Play\\binaural_audios"
video_dir = "H:\\FAIR-Play\\FAIR-Play\\videos"
audio_files = os.listdir(audio_dir)
audio_files = [os.path.join(audio_dir, i) for i in audio_files]
video_files = os.listdir(video_dir)
video_files = [os.path.join(video_dir, i) for i in video_files]

segment_length = 0.63
output_root = 'data\\split-2\\'

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
    librosa.output.write_wav(output_root + "audio_seg\\%06d.wav" % (i+1), audio_seg, sr=sr)

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
    if flag:
        cv2.imwrite(output_root + "frame\\%06d.png" % (i+1), frame)
    else:
        print("frame fail: %d" % center_frame_pos)
        print("total frame: %d" % video.get(cv2.CAP_PROP_FRAME_COUNT))
        print("seg: %f - %f" % (start_time, end_time))
        break

    # Slice video
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    # fourcc = cv2.VideoWriter_fourcc(*'H264')
    out_seg = cv2.VideoWriter(output_root + "video_seg\\%06d.mp4" % (i+1), fourcc, fps, size)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for f in range(start_frame, end_frame+1):
        success, frame = video.read()
        if success:
            out_seg.write(frame)
        else:
            print("read fail: %d\n" % f)
            print("total frame: %d" % video.get(cv2.CAP_PROP_FRAME_COUNT))
            exit()
    video.release()
    out_seg.release()

