import os
import argparse
import librosa
import cv2
import numpy as np
from PIL import Image
import subprocess
from options.test_options import TestOptions
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataloader
import torch

def stft_distance(predicted_audio, gt_audio):
    dis = (predicted_audio[:,0] - gt_audio[:,0])**2 + (predicted_audio[:,1] - gt_audio[:,1])**2
    return dis

def env_distance(predicted_audio, gt_audio):
    

def main():

    opt = TestOptions().parse()
    device = torch.device("cuda:0")
    opt.mode = 'tset'

    #Contruct validation dataset
    dataset = CustomDataset(opt)
    dataloader = Dataloader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.nThreads)
    )

    #load model
    model = torch.load(opt.model_path)
    model.to(device)

    #initialize 


    #predict output


    #recontruct binaural audio

    #compute STFT-distance & ENV-distance


if __name__ == "__main__":
    main()