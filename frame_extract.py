from PIL import Image, ImageEnhance
import h5py
from tqdm import tqdm
import os
import torch
import numpy as np
import random

frame_dir = 'data\\split-2\\frame'
h5_path = 'data\\split-2\\frame.h5'

mean = [0.41830324, 0.35137988, 0.31830999]
std = [0.26464085, 0.22256009, 0.23320136]

def mian():

    files = os.listdir(frame_dir)
    frame_path = [os.path.join(frame_dir, f) for f in files]

    frame_list = np.empty(shape=(0,224,448,3))

    with h5py.File(h5_path, 'w') as f:

        for index in tqdm(range(len(frame_path))):
            image = Image.open(frame_path[index])
            image = image.resize((448,224))
            #enhance
            # bright_enhancer = ImageEnhance.Brightness(image)
            # image = bright_enhancer.enhance(random.random()*0.6 + 0.7)
            # color_enhancer = ImageEnhance.Color(image)
            # image = color_enhancer.enhance(random.random()*0.6 + 0.7)
            #normalization
            image_array = np.array(image) / 255
            image_array = (image_array - mean) / std

            frame_list = np.concatenate([frame_list, [image_array]])
            
            if index == 0:
                dataset = f.create_dataset('frame', data=frame_list, chunks=True, maxshape=(None,224,448,3))
                frame_list = np.empty(shape=(0,224,448,3))

            elif index % 100 == 0 or index == len(frame_path)-1:
                dataset.resize(dataset.shape[0] + len(frame_list), axis=0)
                dataset[-len(frame_list):] = frame_list
                frame_list = np.empty(shape=(0,224,448,3))
                print(len(dataset))


if __name__ == '__main__':
    mian()