import h5py
from tqdm import tqdm
import os
import random
import numpy as np

frame_path = 'data\\split-2\\frame.h5'
diff_spec_path = 'data\\split-2\\audio_diff_spec.h5'
mix_spec_path = 'data\\split-2\\audio_mix_spec.h5'
output_path = 'data\\split-3\\'

def main():

    frame = h5py.File(frame_path)
    diff_spec = h5py.File(diff_spec_path)
    mix_spec = h5py.File(mix_spec_path)

    with h5py.File(output_path+'train.h5', 'w') as f:
        frame_dset = f.create_dataset('frame', data=frame['frame'][:1497])
        frame_dset = f.create_dataset('audio_diff', data=diff_spec['audio_diff_spec'][:1497])
        frame_dset = f.create_dataset('audio_mix', data=mix_spec['audio_mix_spec'][:1497])

    with h5py.File(output_path+'test.h5', 'w') as f:
        frame_dset = f.create_dataset('frame', data=frame['frame'][1497:1684])
        frame_dset = f.create_dataset('audio_diff', data=diff_spec['audio_diff_spec'][1497:1684])
        frame_dset = f.create_dataset('audio_mix', data=mix_spec['audio_mix_spec'][1497:1684])

    with h5py.File(output_path+'val.h5', 'w') as f:
        frame_dset = f.create_dataset('frame', data=frame['frame'][1684:])
        frame_dset = f.create_dataset('audio_diff', data=diff_spec['audio_diff_spec'][1684:])
        frame_dset = f.create_dataset('audio_mix', data=mix_spec['audio_mix_spec'][1684:])


if __name__ == '__main__':
    main()