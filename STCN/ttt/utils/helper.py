import os
import time
import torch
import numpy as np
import torch.nn.functional as F

from os import path
from PIL import Image
from inference_core import InferenceCore as InferenceCoreDavis
from inference_core_yv import InferenceCore as InferenceCoreYT
from util.tensor_util import unpad

from dataset.davis_test_dataset import DAVISTestDataset
from dataset.yv_test_dataset import YouTubeVOSTestDataset
from dataset.mose_test_dataset import MOSETestDataset
from model.eval_network import STCN


def worker_init_fn(worker_id):
    return np.random.seed(torch.initial_seed() % (2**31) + worker_id)

@torch.no_grad()
def video_inference(data, prop_model, output_dir, args):
    if args.dataset_name == "youtube":
        video_inference_youtube(data, prop_model, output_dir, args)
    else:
        video_inference_not_youtube(data, prop_model, output_dir, args)


@torch.no_grad()
def video_inference_youtube(data, prop_model, output_dir, args):
    with torch.cuda.amp.autocast(enabled=args.amp):
        rgb = data['rgb']
        msk = data['gt'][0]
        info = data['info']
        name = info['name'][0]
        num_objects = len(info['labels'][0])  # (k in davis)
        gt_obj = info['gt_obj']  # not in davis
        size = info['size']

        # Frames with labels, but they are not exhaustively labeled
        frames_with_gt = sorted(list(gt_obj.keys()))

        # torch.cuda.synchronize()
        process_begin = time.time()

        processor = InferenceCoreYT(prop_model, rgb, num_objects, top_k=args.top,
                                  mem_every=args.mem_every, include_last=args.include_last)

        # min_idx tells us the starting point of propagation
        # Propagating before there are labels is not useful
        min_idx = 99999

        for i, frame_idx in enumerate(frames_with_gt):
            min_idx = min(frame_idx, min_idx)
            # Note that there might be more than one label per frame
            obj_idx = gt_obj[frame_idx][0].tolist()
            # Map the possibly non-continuous labels into a continuous scheme
            obj_idx = [info['label_convert'][o].item() for o in obj_idx]

            # Append the background label
            with_bg_msk = torch.cat([
                1 - torch.sum(msk[:, frame_idx], dim=0, keepdim=True),
                msk[:, frame_idx],
            ], 0).cuda()

            # We perform propagation from the current frame to the next frame with label
            if i == len(frames_with_gt) - 1:
                processor.interact(with_bg_msk, frame_idx, rgb.shape[1], obj_idx)
            else:
                processor.interact(with_bg_msk, frame_idx, frames_with_gt[i + 1] + 1, obj_idx)

        # Do unpad -> upsample to original size (we made it 480p)
        out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
        for ti in range(processor.t):
            prob = unpad(processor.prob[:, ti], processor.pad)
            prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
            out_masks[ti] = torch.argmax(prob, dim=0)

        out_masks = (out_masks.detach().cpu().numpy()[:,0]).astype(np.uint8)

        # Remap the indices to the original domain
        idx_masks = np.zeros_like(out_masks)
        for i in range(1, num_objects+1):
            backward_idx = info['label_backward'][i].item()
            idx_masks[out_masks==i] = backward_idx
        stats = {
            'process_time': time.time() - process_begin,
            'frame_num': out_masks.shape[0]
        }

        # Save the results
        video_output_dir = path.join(output_dir, name)
        os.makedirs(video_output_dir, exist_ok=True)
        for f in range(idx_masks.shape[0]):
            if f >= min_idx:
                # if args.output_all or (f in req_frames):
                img_E = Image.fromarray(idx_masks[f])
                img_E.putpalette(args.palette)
                img_E.save(os.path.join(video_output_dir, info['frames'][f][0].replace('.jpg','.png')))

        return processor.prob, stats


@torch.no_grad()
def video_inference_not_youtube(data, prop_model, output_dir, args):
    with torch.cuda.amp.autocast(enabled=args.amp):
        rgb = data['rgb'].cuda()
        msk = data['gt'][0].cuda()
        info = data['info']
        name = info['name'][0]
        k = len(info['labels'][0])
        size = info['size_480p']

        # torch.cuda.synchronize()
        process_begin = time.time()

        processor = InferenceCoreDavis(
            prop_model, rgb, k, top_k=args.top, mem_every=args.mem_every, include_last=args.include_last)
        processor.interact(msk[:, 0], 0, rgb.shape[1])

        # Do unpad -> upsample to original size
        out_masks = torch.zeros((processor.t, 1, *size), dtype=torch.uint8, device='cuda')
        for ti in range(processor.t):
            prob = unpad(processor.prob[:, ti], processor.pad)
            prob = F.interpolate(prob, size, mode='bilinear', align_corners=False)
            out_masks[ti] = torch.argmax(prob, dim=0)

        out_masks = (out_masks.detach().cpu().numpy()[:, 0]).astype(np.uint8)

        stats = {
            'process_time': time.time() - process_begin,
            'frame_num': out_masks.shape[0]
        }

        if output_dir is not None:
            # Save the results
            video_output_dir = path.join(output_dir, name)
            os.makedirs(video_output_dir, exist_ok=True)
            for f in range(out_masks.shape[0]):
                img_E = Image.fromarray(out_masks[f])
                img_E.putpalette(args.palette)
                img_E.save(os.path.join(video_output_dir, '{:05d}.png'.format(f)))

        return processor.prob, stats


def get_frame_dirs(args):
    if args.corrupted_image_dir is None:
        if args.dataset_name == "youtube":
            frame_dir = path.join(args.dataset_dir, args.split, "JPEGImages")
            all_frames_dir = path.join(args.dataset_dir, 'all_frames', 'valid_all_frames', 'JPEGImages')
        elif args.dataset_name == "davis":
            frame_dir = path.join(args.dataset_dir, 'trainval', 'JPEGImages', '480p')
            all_frames_dir = None
        elif args.dataset_name == "mose":
            frame_dir = path.join(args.dataset_dir, args.split, 'JPEGImages')
            all_frames_dir = None
    else:
        frame_dir = args.corrupted_image_dir
        all_frames_dir = frame_dir if args.dataset_name == "youtube" else None
    return frame_dir, all_frames_dir


def dump_logs(output_file, logs):
    with open(output_file, "w") as writer:
        for l in logs:
            writer.write(';'.join(list(map(str, l))) + "\n")

def get_palette(args):
    if args.dataset_name == "youtube":
        palette_video_name = "0a49f5265b" if args.palette_video_name is None else args.palette_video_name
        palette_filename = args.dataset_dir + f'/valid/Annotations/{palette_video_name}/00000.png'
    elif args.dataset_name.startswith("davis"):
        palette_video_name = "blackswan" if args.palette_video_name is None else args.palette_video_name
        palette_filename = args.dataset_dir + f'/trainval/Annotations/480p/{palette_video_name}/00000.png'
    elif args.dataset_name == "mose":
        palette_video_name = "009ddff6" if args.palette_video_name is None else args.palette_video_name
        palette_filename = args.dataset_dir + f'/{args.split}/Annotations/{palette_video_name}/00000.png'
    palette = Image.open(path.expanduser(palette_filename)).getpalette()
    return palette


def get_test_dataset(args):
    # Filter video names
    video_names = None
    if args.video_set_filename is not None:
        video_names = sorted(np.loadtxt(args.video_set_filename, dtype=str))

    # Get test dataset
    if args.dataset_name == "youtube":
        video_names = video_names[:25]
        test_dataset = YouTubeVOSTestDataset(data_root=args.dataset_dir, split=args.split, video_ids=video_names)
    elif args.dataset_name.startswith("davis"):
        if args.split == 'val':
            test_dataset = DAVISTestDataset(args.dataset_dir + '/trainval', imset='2017/val.txt',
                                            corrupt_dir=args.corrupted_image_dir)
        elif args.split == 'testdev':
            test_dataset = DAVISTestDataset(args.dataset_dir + '/test-dev', imset='2017/test-dev.txt')
        else:
            raise NotImplementedError
    elif args.dataset_name == "mose":
        test_dataset = MOSETestDataset(path.join(args.dataset_dir, args.split), video_ids=video_names)
    return test_dataset


def get_stcn_model(args):
    stcn_model = STCN().cuda().eval()
    # Performs input mapping such that stage 0 model can be loaded
    prop_saved = torch.load(args.model_filename)
    for k in list(prop_saved.keys()):
        if k == 'value_encoder.conv1.weight':
            if prop_saved[k].shape[1] == 4:
                pads = torch.zeros((64, 1, 7, 7), device=prop_saved[k].device)
                prop_saved[k] = torch.cat([prop_saved[k], pads], 1)
    stcn_model.load_state_dict(prop_saved)
    return stcn_model
