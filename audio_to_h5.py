import librosa
import h5py
import os
from tqdm import tqdm
import numpy as np

ADUIO_DIR = "H:\\FAIR-Play\\FAIR-Play\\binaural_audios"
OUTPUT_DIR = "H:\\FAIR-Play\\FAIR-Play\\audio_h5\\"
SAMPLE_RATE = 16000
AUDIO_LEN = 9

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
    files = os.listdir(ADUIO_DIR)
    files = [os.path.join(ADUIO_DIR, i) for i in files]

    audio_list = np.empty((0,2,SAMPLE_RATE*AUDIO_LEN))

    for i in tqdm(range(len(files)), ascii=True):
        audio, sr = librosa.load(files[i], sr=SAMPLE_RATE, mono=False)
        audio = audio[:, :AUDIO_LEN*SAMPLE_RATE]

        audio_list = np.concatenate([audio_list, [audio]])

        if (i+1)%100 == 0 or i == len(files)-1:
            write_hdf5('audio.h5', audio_list)
            audio_list = np.empty((0,2,SAMPLE_RATE*AUDIO_LEN))


if __name__ == "__main__":
    main()