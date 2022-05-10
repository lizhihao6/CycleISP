"""
 Copyright 2020 Mahmoud Afifi.
 Released under the MIT License.
 If you use this code, please cite the following paper:
 Mahmoud Afifi, Abdelrahman Abdelhamed, Abdullah Abuolaim, Abhijith
 Punnappurath, and Michael S Brown.
 CIE XYZ Net: Unprocessing Images for Low-Level Computer Vision Tasks.
 arXiv preprint, 2020.
"""

__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

from os.path import join
from os import listdir
from os import path
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import cv2
import rawpy

IPHONE_PATH = '/lzh/datasets/multiRAW/iphone_xsmax'

class BasicDataset(Dataset):
    def __init__(self, patch_size=256, val=False):
        split = join(IPHONE_PATH, 'train.txt') if not val else join(IPHONE_PATH, 'test.txt')
        with open(split, 'r') as f:
            self.raws = [join(IPHONE_PATH, 'raw', f'{line.strip()}.DNG') for line in f]
        self.rgbs = [r.replace('raw', 'camera_isp').replace('.DNG', '.png') for r in self.raws]
        self.patch_size = patch_size
        logging.info(f'Creating dataset with {len(self.raws)} examples')

    def __len__(self):
        return len(self.raws)

    @classmethod
    def preprocess(cls, img, patch_size, w, h, patch_coords, aug_op, scale=1):
        if aug_op is 1:
            img = cv2.flip(img, 0)
        elif aug_op is 2:
            img = cv2.flip(img, 1)
        elif aug_op is 3:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        img_nd = np.array(img)
        assert len(img_nd.shape) == 3, 'Training/validation images ' \
                                       'should be 3 channels colored images'
        img_nd = img_nd[patch_coords[1]:patch_coords[1]+patch_size,
                 patch_coords[0]:patch_coords[0]+patch_size, :]
        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans

    def __getitem__(self, i):
        raw, rgb = self.raws[i], self.rgbs[i]
        with rawpy.imread(raw) as f:
            raw = f.raw_image_visible.copy().astype(np.float32)
        raw = (raw-528)/(4095-528)
        raw = np.clip(raw, 0, 1)        
        h, w = raw.shape

        demo = np.zeros((h, w, 3), dtype=raw.dtype)
        raw = np.pad(raw, ((2, 2), (2, 2)), mode='reflect')
        r, gb, gr, b = raw[0::2, 0::2], raw[0::2, 1::2], raw[1::2, 0::2], raw[1::2, 1::2]
        
        demo[0::2, 0::2, 0] = r[1:-1, 1:-1]
        demo[0::2, 0::2, 1] = (gr[1:-1, 1:-1] + gr[:-2, 1:-1] + gb[1:-1, 1:-1] + gb[1:-1, :-2]) / 4
        demo[0::2, 0::2, 2] = (b[1:-1, 1:-1] + b[:-2, :-2] + b[1:-1, :-2] + b[:-2, 1:-1]) / 4

        demo[1::2, 0::2, 0] = (r[1:-1, 1:-1] + r[2:, 1:-1]) / 2
        demo[1::2, 0::2, 1] = gr[1:-1, 1:-1]
        demo[1::2, 0::2, 2] = (b[1:-1, 1:-1] + b[1:-1, :-2]) / 2
        
        demo[0::2, 1::2, 0] = (r[1:-1, 1:-1] + r[1:-1, 2:]) / 2
        demo[0::2, 1::2, 1] = gb[1:-1, 1:-1]
        demo[0::2, 1::2, 2] = (b[1:-1, 1:-1] + b[:-2, 1:-1]) / 2
        
        demo[1::2, 1::2, 0] = (r[1:-1, 1:-1] + r[2:, 2:] + r[1:-1, 2:] + r[2:, 1:-1]) / 4
        demo[1::2, 1::2, 1] = (gr[1:-1, 1:-1] + gr[1:-1, 2:] + gb[1:-1, 1:-1] + gb[2:, 1:-1]) / 4
        demo[1::2, 1::2, 2] = b[1:-1, 1:-1]
        raw = demo

        rgb = cv2.imread(rgb).astype(np.float32)[..., ::-1]/255


        # get augmentation option
        aug_op = np.random.randint(4)
        if aug_op == 3:
            scale = np.random.uniform(low=1.0, high=1.2)
        else:
            scale = 1
        # get random patch coord
        patch_x = np.random.randint(0, high=w - self.patch_size)
        patch_y = np.random.randint(0, high=h - self.patch_size)
        rgb = self.preprocess(rgb, self.patch_size, w, h, (
            patch_x, patch_y), aug_op, scale=scale)
        raw = self.preprocess(raw, self.patch_size, w, h, (
            patch_x, patch_y), aug_op, scale=scale)

        return {'image': torch.from_numpy(rgb), 'gt_xyz':
            torch.from_numpy(raw)}
