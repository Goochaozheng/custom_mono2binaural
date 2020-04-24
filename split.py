import h5py
from tqdm import tqdm
import os
import random
import numpy as np

frame_path = 'data\\split-6\\frame.h5'
diff_spec_path = 'data\\split-6\\audio_diff_spec.h5'
mix_spec_path = 'data\\split-6\\audio_mix_spec.h5'
output_path = 'data\\split-6\\'

def main():

    frame = h5py.File(frame_path)
    diff_spec = h5py.File(diff_spec_path)
    mix_spec = h5py.File(mix_spec_path)

    with h5py.File(output_path+'train.h5', 'w') as f:
        f.create_dataset('frame', data=frame['frame'][:4500])
        f.create_dataset('audio_diff', data=diff_spec['audio_diff_spec'][:4500])
        f.create_dataset('audio_mix', data=mix_spec['audio_mix_spec'][:4500])

    with h5py.File(output_path+'test.h5', 'w') as f:
        f.create_dataset('frame', data=frame['frame'][4500:4750])
        f.create_dataset('audio_diff', data=diff_spec['audio_diff_spec'][4500:4750])
        f.create_dataset('audio_mix', data=mix_spec['audio_mix_spec'][4500:4750])

    with h5py.File(output_path+'val.h5', 'w') as f:
        f.create_dataset('frame', data=frame['frame'][4750:])
        f.create_dataset('audio_diff', data=diff_spec['audio_diff_spec'][4750:])
        f.create_dataset('audio_mix', data=mix_spec['audio_mix_spec'][4750:])


if __name__ == '__main__':
    main()