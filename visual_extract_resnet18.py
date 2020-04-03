# Extract visual feature using pretrained resnet-18

import torch
import torchvision
import h5py
from tqdm import tqdm
import os
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import random

class frame_data(Dataset):

    def __init__(self):
        root_dir = "data\\frame"
        frames = os.listdir(root_dir)
        self.frames = [os.path.join(root_dir, i) for i in frames]

    def __augment(self, frame):
        #augment
        enhancer = ImageEnhance.Brightness(frame)
        frame = enhancer.enhance(random.random()*0.6 + 0.7)
        enhancer = ImageEnhance.Color(frame)
        frame = enhancer.enhance(random.random()*0.6 + 0.7)
        return frame

    def __transform(self, frame):
        #transform
        to_tensor = torchvision.transforms.ToTensor()
        norm = torchvision.transforms.Normalize(
            mean=[0.41820913, 0.35141712, 0.31842441],
            std=[0.26459038, 0.22238343, 0.23302009]
        )
        trans = torchvision.transforms.Compose([to_tensor, norm])
        return trans(frame)

    def __process_image(self, image, augment):
        image = image.resize((480,240))
        w,h = image.size
        w_offset = w - 448
        h_offset = h - 224
        left = random.randrange(0, w_offset + 1)
        upper = random.randrange(0, h_offset + 1)
        image = image.crop((left, upper, left+448, upper+224))

        if augment:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.random()*0.6 + 0.7)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(random.random()*0.6 + 0.7)
        return image


    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_data = Image.open(self.frames[idx]).convert('RGB')
        #frame_data = frame_data.resize((448,224))
        frame_data = self.__process_image(frame_data, True)
        frame_data = self.__transform(frame_data)
        return frame_data


def main():

    # Custom dataset
    dataset = frame_data()
    frame_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)

    # Load pre-trained resnet18
    resnet = torchvision.models.resnet18(pretrained=True)
    # Only preserver layers before conv 1x1
    layers = list(resnet.children())[0:-2]
    visual_extraction = torch.nn.Sequential(*layers) 

    writer = SummaryWriter(log_dir='Log/visual_extract_resnet18_cropped')
    writer.add_graph(visual_extraction, next(iter(frame_loader)))

    # Run with GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    visual_extraction.to(device)

    # Write feature to h5py file
    feature_list = []
    for batch_num, frame in enumerate(tqdm(frame_loader)):
        #ResNet extraction
        frame = frame.to(device)
        with torch.no_grad(): 
            feature = visual_extraction(frame)
            feature = feature.cpu()
            feature = feature[0,:,:,:].numpy()
            feature_list.append(feature)

            # Record frame image
            if batch_num%50 == 0:
                writer.add_image('frame', frame[0,:,:,:], batch_num)
    
    # Write to file as numpy array
    feature_list = np.array(feature_list)
    with h5py.File('data/visual_extract_resnet18_cropped.h5', 'w') as f:
        dset = f.create_dataset('features', data=feature_list)

if __name__ == "__main__":
    main()
