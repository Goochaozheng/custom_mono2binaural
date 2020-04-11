import h5py
from tqdm import tqdm
import os
import random
import numpy as np

visual_feature_path = 'data\\split-4\\visual_feature.h5'
diff_spec_path = 'data\\split-4\\audio_diff_spec.h5'
mix_spec_path = 'data\\split-4\\audio_mix_spec.h5'
output_path = 'data\\split-4\\'

def main():

    visual_feature = h5py.File(visual_feature_path)
    diff_spec = h5py.File(diff_spec_path)
    mix_spec = h5py.File(mix_spec_path)

    with h5py.File(output_path+'train.h5', 'w') as f:
        f.create_dataset('visual_feature', data=visual_feature['visual_feature'][:4500])
        f.create_dataset('audio_diff', data=diff_spec['audio_diff_spec'][:4500])
        f.create_dataset('audio_mix', data=mix_spec['audio_mix_spec'][:4500])

    with h5py.File(output_path+'test.h5', 'w') as f:
        f.create_dataset('visual_feature', data=visual_feature['visual_feature'][4500:4750])
        f.create_dataset('audio_diff', data=diff_spec['audio_diff_spec'][4500:4750])
        f.create_dataset('audio_mix', data=mix_spec['audio_mix_spec'][4500:4750])

    with h5py.File(output_path+'val.h5', 'w') as f:
        f.create_dataset('visual_feature', data=visual_feature['visual_feature'][4750:])
        f.create_dataset('audio_diff', data=diff_spec['audio_diff_spec'][4750:])
        f.create_dataset('audio_mix', data=mix_spec['audio_mix_spec'][4750:])


if __name__ == '__main__':
    main()