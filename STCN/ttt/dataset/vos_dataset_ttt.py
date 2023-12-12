import copy
import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed
from dataset.util import all_to_onehot
from util.tensor_util import pad_divide_by


class VOSDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self,
                 im_root,
                 gt_root,
                 video_name,
                 max_jump,
                 num_frames,
                 total_sequences=1,
                 video_percentage=1.,
                 mem_every=None,
                 resolution=480,
                 scale=(1., 1.),
                 ratio=(1., 1.),
                 augmentations='none',
                 coverage=0.,
                 all_objects=False,
                 check_last=True,
                 max_obj=6,
                 frames_with_gt=[0],
                 im_root_all_frames=None  # needed for yt
                 ):

        self.im_root = path.join(im_root, video_name)
        self.gt_root = path.join(gt_root, video_name)
        self.video = video_name

        if im_root_all_frames is None:
            self.im_root_all_frames = self.im_root
            self.frames = sorted(os.listdir(self.im_root))
            mask = Image.open(path.join(self.gt_root, self.frames[0][:-4] + '.png')).convert('P')
            self.all_labels = np.unique(np.array(mask))
            self.relative_position_first_frame = 0
        else:
            self.im_root_all_frames = path.join(im_root_all_frames, video_name)
            subsampled_frames = sorted(os.listdir(self.im_root))
            self.all_frames = sorted(os.listdir(self.im_root_all_frames))
            first_subsampled_frame_id = int(subsampled_frames[0].split(".")[0])
            self.frames = [(i_frame, frame) for i_frame, frame in enumerate(self.all_frames) if
                           int(frame.split(".")[0]) >= first_subsampled_frame_id]
            self.relative_position_first_frame = int(self.frames[0][0])
            self.frames = [frame for i_frame, frame in self.frames]

        self.frames_with_gt = frames_with_gt  # this is needed for the yt dataset
        self.total_sequences = total_sequences
        self.len_frames = int(len(self.frames) * video_percentage)
        self.check_last = check_last
        self.max_obj = max_obj
        self.max_jump = max_jump
        self.num_frames = num_frames
        self.mem_every = mem_every
        self.all_objects = all_objects
        self.coverage = coverage

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ]) if 'colour' in augmentations else lambda x: x

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
        ]) if 'geometric' in augmentations else lambda x: x

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
        ]) if 'geometric' in augmentations else lambda x: x

        # These transforms are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ]) if 'colour' in augmentations else lambda x: x

        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip() if 'geometric' in augmentations else transforms.Lambda(lambda x: x),
            transforms.RandomResizedCrop((resolution, resolution), scale=scale, ratio=ratio,
                                         interpolation=Image.BICUBIC)
            if ratio != (1., 1.) or scale != (1., 1.) else
            transforms.Resize(resolution, interpolation=Image.BICUBIC)
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip() if 'geometric' in augmentations else transforms.Lambda(lambda x: x),
            transforms.RandomResizedCrop((resolution, resolution), scale=scale, ratio=ratio,
                                         interpolation=Image.NEAREST) if ratio != (1., 1.) or scale != (1., 1.)
            else transforms.Resize(resolution, interpolation=Image.NEAREST)
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])


    def select_first_frame_id(self):
        """ Selects the id of the first frame of the triplet. In DAVIS, DAVIS-C and Mose, it is always the first
        frame but for the YouTube dataset, it might be a later frame. Additionally, in the youtube dataset, each
        object may be annotated for the first time in a different frame (represented in self.frames_with_gt).
        """
        frame_with_gt = np.random.choice(self.frames_with_gt)
        first_frame_id = (frame_with_gt - self.relative_position_first_frame)
        filename = path.join(self.gt_root, self.frames[first_frame_id][:-4] + '.png')
        mask = Image.open(filename).convert('P')
        all_labels = np.unique(np.array(mask))
        return first_frame_id, all_labels

    def get_sequence(self, first_frame_id):
        if self.num_frames == 1:
            return [first_frame_id]

        # Don't want to bias towards beginning/end
        if isinstance(self.max_jump, tuple):
            this_max_jump = np.random.randint(*self.max_jump)
        else:
            this_max_jump = self.max_jump
        this_max_jump = min(self.len_frames - first_frame_id - 1, this_max_jump)

        frames_idx = [first_frame_id]
        for i in range(self.num_frames - 2):
            if self.mem_every is not None:
                r = np.arange(self.mem_every, self.len_frames, self.mem_every)
                r = r[np.logical_and(r > frames_idx[-1], r <= frames_idx[-1] + this_max_jump)]
                f_idx = np.random.choice(r) if len(r) else frames_idx[-1] + self.mem_every
            else:
                f_idx = frames_idx[-1] + np.random.randint(this_max_jump) + 1
            f_idx = min(f_idx, self.len_frames - this_max_jump, self.len_frames - 1)
            frames_idx.append(f_idx)

        f_idx = frames_idx[-1] + np.random.randint(this_max_jump) + 1
        f_idx = min(f_idx, self.len_frames - 1)
        frames_idx.append(f_idx)

        return frames_idx

    def __getitem__(self, idx):
        info = {'name': self.video}

        trials, limit = 0, 100
        while trials < limit:
            while True:
                first_frame_id, all_labels = self.select_first_frame_id()
                frames_idx = self.get_sequence(first_frame_id)
                if len(frames_idx) == 1:  # for the tt-AE baseline, no triplet is used
                    break
                if frames_idx[-1] != frames_idx[-2]:
                    break

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_object = None
            for f_idx in frames_idx:
                jpg_name = self.frames[f_idx][:-4] + '.jpg'
                png_name = self.frames[f_idx][:-4] + '.png'

                reseed(sequence_seed)
                filename_img = path.join(self.im_root_all_frames, jpg_name)
                this_im = Image.open(filename_img).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)

                reseed(sequence_seed)
                mask_name = path.join(self.gt_root, png_name)
                this_gt = Image.open(mask_name).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels != 0]

            if len(labels) == 0:
                target_object = -1  # all black if no objects
                has_second_object = False
            else:
                if self.all_objects:
                    target_objects = np.random.choice(labels, np.minimum(len(labels), self.max_obj), replace=False)
                    if not self.check_last or len(set(target_objects) - set(np.unique(masks[-1]))) == 0:
                        break

                    if len(all_labels) == len(np.unique(masks[0])) == len(np.unique(masks[-1])):
                        break
                    if trials > limit // 2 and len(all_labels) == len(np.unique(masks[0])):
                        break
                else:
                    target_object = np.random.choice(labels)
                    has_second_object = (len(labels) > 1)
                    if has_second_object:
                        second_object = np.random.choice(labels[labels != target_object])
                    ratio = (masks[-1] == target_object).mean() / (masks[0] == target_object).mean()
                    if self.coverage <= 0. or ratio > self.coverage:
                        break
            trials += 1

        if self.check_last and (
                self.all_objects and len(np.unique(masks[0])) != len(np.unique(masks[-1]))) or trials >= limit:
            images[-1] = copy.deepcopy(images[0])
            masks[-1] = copy.deepcopy(masks[0])
            frames_idx[-1] = first_frame_id
        info['frames_idx'] = frames_idx

        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)

        images = pad_divide_by(images, 16)[0]
        masks = pad_divide_by(torch.from_numpy(masks), 16)[0].numpy()

        if self.all_objects:
            labels = np.unique(masks[0])
            labels = labels[labels != 0]

            cls_gt = np.zeros(masks.shape, dtype=np.int64)
            for i, l in enumerate(labels):
                cls_gt[masks == l] = i + 1

            obj_masks = torch.from_numpy(all_to_onehot(cls_gt, labels)).float()
            obj_masks = obj_masks.unsqueeze(2)  # O x T x 1 x H x W

            object_count = obj_masks.shape[0]
            if object_count > 1:
                other_mask = torch.sum(obj_masks, dim=0, keepdim=True) - obj_masks
                selector = torch.FloatTensor([1 for _ in range(object_count)])
                if len(target_objects) < len(labels):
                    obj_masks = obj_masks[(target_objects - 1).tolist()]
                    other_mask = other_mask[(target_objects - 1).tolist()]
                    selector = selector[(target_objects - 1).tolist()]
                    cls_gt = np.zeros(masks.shape, dtype=np.int64)
                    for i, l in enumerate(target_objects):
                        cls_gt[masks == l] = i + 1
            else:
                other_mask = torch.cat([torch.zeros_like(obj_masks), obj_masks], 0)
                obj_masks = torch.cat([obj_masks, torch.zeros_like(obj_masks)], 0)
                selector = torch.FloatTensor([1, 0])
        else:
            tar_masks = (masks == target_object).astype(np.float32)[:, None, :, :]
            if has_second_object:
                sec_masks = (masks == second_object).astype(np.float32)[:, None, :, :]
                selector = torch.FloatTensor([1, 1])
            else:
                sec_masks = np.zeros_like(tar_masks)
                selector = torch.FloatTensor([1, 0])

            obj_masks = np.stack([tar_masks, sec_masks])
            other_mask = np.stack([sec_masks, tar_masks])
            cls_gt = np.zeros(masks.shape, dtype=np.int64)
            cls_gt[tar_masks[:, 0] > 0.5] = 1
            cls_gt[sec_masks[:, 0] > 0.5] = 2

        labels = np.unique(masks[0])
        labels = labels[labels != 0]
        info['labels'] = labels

        data = {
            'rgb': images,
            'gt': obj_masks,
            'cls_gt': cls_gt,
            'sec_gt': other_mask,
            'selector': selector,
            'info': info,
        }
        return data

    def __len__(self):
        return self.total_sequences
