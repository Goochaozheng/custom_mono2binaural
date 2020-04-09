import h5py
from tqdm import tqdm
import os
import random
import numpy as np

visual_feature_path = 'data\\split-2\\visual_extract_resnet18.h5'
diff_spec_path = 'data\\split-2\\audio_diff_spec.h5'
mix_spec_path = 'data\\split-2\\audio_mix_spec.h5'
output_path = 'data\\split-2\\'

def create_h5py(mode, range_begin, range_end, data):

    print('Create h5py for ', mode)

    temp_visual = []
    temp_diff = []
    temp_mix = []

    for i in tqdm(range(range_begin, range_end)):
        temp_visual.append(data[i][0])
        temp_diff.append(data[i][1])
        temp_mix.append(data[i][2])

    temp_visual = np.array(temp_visual)
    temp_diff = np.array(temp_diff)
    temp_mix = np.array(temp_mix)

    with h5py.File(output_path+mode+'.h5', 'w') as f:
        f.create_dataset('visual_feature', data=temp_visual)
        f.create_dataset('audio_diff', data=temp_diff)
        f.create_dataset('audio_mix', data=temp_mix)


def main():

    visual_feature = h5py.File(visual_feature_path)
    diff_spec = h5py.File(diff_spec_path)
    mix_spec = h5py.File(mix_spec_path)

    data = []

    for i in tqdm(range(len(visual_feature['features']))):
        item = [
            visual_feature['features'][i],
            diff_spec['audio_diff_spec'][i],
            mix_spec['audio_mix_spec'][i]
        ]
        data.append(item)

    random.shuffle(data)

    create_h5py(mode='train', range_begin=0, range_end=1497, data=data)
    create_h5py(mode='test', range_begin=1497, range_end=1684, data=data)
    create_h5py(mode='val', range_begin=1684, range_end=1871, data=data)

if __name__ == '__main__':
    main()