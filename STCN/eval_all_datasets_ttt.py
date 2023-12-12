import random
import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
from torch.utils.data import DataLoader

from model.eval_network import STCN
from model.losses import EntropyLoss

from ttt.config.load_config import load_config
from ttt.model.model_ttt import STCN_TTT
from ttt.utils.meter import AverageMeterDict
from ttt.utils.helper import *
from ttt.dataset.vos_dataset_ttt import VOSDataset


def test_time_train_and_evaluate_one_video(args, video_data, pretrained_model):
    """
    For a given video, runs the test time training process which updates the weights of the pre-trained model.
    Then evaluates the updated model on the given video.
    """
    video_name = video_data['info']['name'][0]

    # Fix the seed for this video
    seed = args.seed
    output_seed_dir = os.path.join(args.output_dir, str(seed))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize the training and inference models
    test_time_training_model = STCN_TTT().cuda().eval()
    test_time_inference_model = STCN().cuda().eval()

    weights = pretrained_model.state_dict()

    print('\nvideo:', video_name, 'shape', video_data['rgb'][0].shape, 'objects:', len(video_data['info']['labels'][0]))
    if args.ttt_number_iterations_per_jump_step > 0:
        test_time_training_model.copy_weights_from(weights)
        test_time_training_model.freeze_parse(args.ttt_frozen_layers)

        result_dir = os.path.join(output_seed_dir, 'final', video_name)
        if not os.path.exists(result_dir) or not len(os.listdir(result_dir)) or args.overwrite:

            # Run test time training
            logs, ttt_models = test_time_train_one_video(
                test_time_training_model, video_name, video_data, output_seed_dir, args)

            # Run test time inference
            test_time_evaluate_one_video(ttt_models, output_seed_dir, test_time_inference_model, video_data)

            # Save in logs
            if logs is not None:
                log_dir = os.path.join(output_seed_dir, 'logs')
                os.makedirs(log_dir, exist_ok=True)
                dump_logs(os.path.join(log_dir, video_name + '.txt'), logs)


def test_time_train_one_video(model, video_name, vid_reader, result_dir, args):
    """ For a given video, runs the test time training process which updates the weights of the pre-trained model. """
    ce_criterion = nn.CrossEntropyLoss()
    ent_criterion = EntropyLoss(dim=1)

    val_model = STCN().cuda().eval()
    model.copy_weights_to(val_model)
    video_inference(vid_reader, val_model, os.path.join(result_dir, 'temp'), args)

    # Parameters for VOSDataset
    frames_with_gt = sorted(list(vid_reader['info']['gt_obj'].keys())) if args.dataset_name == "youtube" else [0]
    max_obj = 6
    all_objects = len(frames_with_gt) == 1
    frame_dir, all_frames_dir = get_frame_dirs(args)

    iteration, ttt_models, logs = 0, dict(), []
    for _ in range(args.ttt_number_jump_steps):
        for max_jump, num_frames in zip(args.ttt_max_jump_step, args.ttt_sequence_length):

            # Evaluate the current model and save the results in the temp folder
            if args.ttt_loss == "tt-mcc":
                model.copy_weights_to(val_model)
                video_inference(vid_reader, val_model, os.path.join(result_dir, 'temp'), args)

            dataset = VOSDataset(frame_dir,
                                 os.path.join(result_dir, 'temp'),
                                 video_name,
                                 max_jump,
                                 num_frames,
                                 total_sequences=args.ttt_number_iterations_per_jump_step * args.ttt_batch_size,
                                 resolution=args.ttt_resolution,
                                 # scale=args.ttt_scale,
                                 # ratio=args.ttt_ratio,
                                 augmentations=args.ttt_augmentations,
                                 check_last=args.ttt_loss == "tt-mcc",
                                 all_objects=all_objects,
                                 max_obj=max_obj,
                                 frames_with_gt=frames_with_gt,
                                 im_root_all_frames=all_frames_dir)
            train_loader = DataLoader(dataset, args.ttt_batch_size, num_workers=16, pin_memory=True,
                                      worker_init_fn=worker_init_fn)

            optimizer = torch.optim.Adam(filter(
                lambda p: p.requires_grad, model.parameters()), lr=args.ttt_lr, weight_decay=1e-7)
            scaler = torch.cuda.amp.GradScaler()

            meters = AverageMeterDict()
            for data in train_loader:
                optimizer.zero_grad()

                for k, v in data.items():
                    if type(v) != list and type(v) != dict and type(v) != int:
                        data[k] = v.cuda(non_blocking=True)

                with torch.cuda.amp.autocast(enabled=args.amp):

                    if args.ttt_loss == "tt-ae":
                        logits_f, masks_f = model.do_single_pass(data)
                    else:
                        logits_f, logits_b, masks_f, masks_b = model.do_cycle_pass(
                            data, backwards=args.ttt_loss == "tt-mcc", encode_first=False)

                    # Loss
                    if args.ttt_loss == "tt-mcc":  # Mask Cycle Consistency
                        loss = ce_criterion(logits_b[-1], data['cls_gt'][:, 0])
                    elif args.ttt_loss == "tt-ae":  # Auto Encoder
                        loss = ce_criterion(logits_f[0], data['cls_gt'][:, 0])
                    elif args.ttt_loss == "tt-ent":  # Entropy
                        loss = ent_criterion(torch.cat(logits_f, 0))

                meters.update('loss', loss)

                if args.amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                iteration += 1

    ttt_models['final'] = model.state_dict()
    model.copy_weights_to(val_model)

    return logs, ttt_models


def test_time_evaluate_one_video(ttt_models, output_seed_dir, test_time_inference_model, data):
    for k, v in ttt_models.items():
        result_dir = os.path.join(output_seed_dir, k)
        os.makedirs(result_dir, exist_ok=True)
        test_time_inference_model.load_state_dict(v)
        video_inference(data, test_time_inference_model, result_dir, args)

def get_parameters():
    args = load_config()

    if len(args.ttt_max_jump_step) > 1 and len(args.ttt_sequence_length) == 1:
        args.ttt_sequence_length = args.ttt_sequence_length * len(args.ttt_max_jump_step)
    elif len(args.ttt_sequence_length) > 1 and len(args.ttt_max_jump_step) == 1:
        args.ttt_max_jump_step = len(args.ttt_sequence_length) * args.ttt_max_jump_step
    elif len(args.ttt_max_jump_step) != len(args.ttt_sequence_length):
        raise Exception('ttt_max_jump_step and ttt_sequence_length should be of equal size or 1.')
    args.palette = get_palette(args)  # load palette
    os.makedirs(args.output_dir, exist_ok=True)  # create the output dir

    print('\nInput Arguments')
    print('---------------')
    for k, v in sorted(dict(vars(args)).items()):
        print('%s: %s' % (k, str(v)))
    print()
    return args


if __name__ == '__main__':
    """
    Arguments loading
    """
    args = get_parameters()

    # Setup Dataset
    test_dataset = get_test_dataset(args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Load our checkpoint
    pretrained_model = get_stcn_model(args)

    for _, video_data in enumerate(test_loader):
        test_time_train_and_evaluate_one_video(args, video_data, pretrained_model)
