"""
Modified from https://github.com/seoungwugoh/STM/blob/master/dataset.py
"""

import os
from os import path
import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data.dataset import Dataset
from dataset.range_transform import im_normalization
from dataset.util import all_to_onehot


class MOSETestDataset(Dataset):
    def __init__(self, root, resolution=480, single_object=False, target_name=None, video_ids=None):
        self.root = root
        self.mask_dir = path.join(root, 'Annotations')
        self.image_dir = path.join(root, 'JPEGImages')
        self.resolution = resolution

        self.videos = []
        self.num_frames = {}
        self.num_objects = {}
        self.shape = {}

        for _video in sorted(os.listdir(self.image_dir)):
            if target_name is not None and target_name != _video:
                continue
            if video_ids is not None and _video not in video_ids:
                continue
            self.videos.append(_video)
            self.num_frames[_video] = len(os.listdir(path.join(self.image_dir, _video)))
            _mask = np.array(Image.open(path.join(self.mask_dir, _video, '00000.png')).convert("P"))
            self.num_objects[_video] = np.max(_mask)
            self.shape[_video] = np.shape(_mask)

        self.single_object = single_object

        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
            transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=InterpolationMode.NEAREST),
        ])

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = self.videos[index]
        info = {}
        info['name'] = video
        info['frames'] = []
        info['num_frames'] = self.num_frames[video]

        images = []
        masks = []

        for f in range(self.num_frames[video]):
            img_file = path.join(self.image_dir, video, '{:05d}.jpg'.format(f))
            images.append(self.im_transform(Image.open(img_file).convert('RGB')))
            info['frames'].append('{:05d}.jpg'.format(f))
            
            mask_file = path.join(self.mask_dir, video, '{:05d}.png'.format(f))
            if path.exists(mask_file):
                masks.append(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8))
            else:
                # Test-set maybe?
                masks.append(np.zeros_like(masks[0]))

        info['size_480p'] = masks[0].shape
        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)
        gt = masks.copy()

        if self.single_object:
            labels = [1]
            masks = (masks > 0.5).astype(np.uint8)
            masks = torch.from_numpy(all_to_onehot(masks, labels)).float()
        else:
            labels = np.unique(masks[0])
            labels = labels[labels!=0]
            masks = torch.from_numpy(all_to_onehot(masks, labels)).float()

        masks = self.mask_transform(masks)
        masks = masks.unsqueeze(2)

        info['size'] = images.shape[2:]
        info['labels'] = labels

        data = {
            'rgb': images,
            'gt': masks,
            'cls_gt': gt,
            'info': info,
        }

        return data
