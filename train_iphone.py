"""
## CycleISP: Real Image Restoration Via Improved Data Synthesis
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## CVPR 2020
## https://arxiv.org/abs/2003.07761
"""

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from networks.cycleisp import Rgb2Raw
import utils
from datasets.iphone import BasicDataset

parser = argparse.ArgumentParser(description='RGB2RAW Network: From clean RGB images, generate {RAW_clean, RAW_noisy} pairs')
parser.add_argument('--input_dir', default='./datasets/sample_rgb_images/',
    type=str, help='Directory of clean RGB images')
parser.add_argument('--result_dir', default='./results/synthesized_data/raw/',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/isp/rgb2raw.pth',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_images', action='store_true', help='Save synthesized images in result directory')

args = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

patchsz = 256
batch_size = 8
train = BasicDataset(patch_size=patchsz, val=False)
val = BasicDataset(patch_size=patchsz, val=True)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
model_rgb2raw = Rgb2Raw()

utils.load_checkpoint(model_rgb2raw,args.weights)
print("===>Testing using weights: ", args.weights)

model_rgb2raw.cuda()

# model_rgb2raw=nn.DataParallel(model_rgb2raw)

# model_rgb2raw.eval()

net = model_rgb2raw
epochs = 100
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)
for epoch in range(epochs):
    net.train()

    epoch_loss = 0
    with tqdm(total=len(train), desc=f'Epoch {epoch + 1}/{epochs}',
                unit='img') as pbar:
        for batch in train_loader:
            imgs = batch['image']
            xyz_gt = batch['gt_xyz']
            assert imgs.shape[1] == 3, (
                    f'Network has been defined with 3 input channels, '
                    f'but loaded training images have {imgs.shape[1]} channels.'
                    f' Please check that the images are loaded correctly.')

            assert xyz_gt.shape[1] == 3, (
                    f'Network has been defined with 3 input channels, '
                    f'but loaded XYZ images have {xyz_gt.shape[1]} channels. '
                    f'Please check that the images are loaded correctly.')

            imgs = imgs.cuda()
            xyz_gt = xyz_gt.cuda()

            raw = net(imgs)
            loss = loss_fn(raw, xyz_gt)
            epoch_loss += loss.item()

            pbar.set_postfix(**{'loss (batch)': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.update(np.ceil(imgs.shape[0]))


    if not os.path.exists('models'):
        os.mkdir('models')
    torch.save(net.state_dict(), 'models/' + 'model_sRGB-XYZ-sRGB.pth')
 