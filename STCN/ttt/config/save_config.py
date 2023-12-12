import argparse
import os
import yaml


def parse_arguments():
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(
        description='This is the training code of a video similarity network based on self-supervision',
        formatter_class=formatter)
    parser.add_argument('--config_name')

    parser.add_argument('--model_filename', default='saves/stcn_s01.pth', help='Path to the pretrained model weights')  # CHANGING
    parser.add_argument('--ttt_lr', default=1e-5, type=float, help='Learning rate use for TTT')  # CHANGING

    # Global parameters
    parser.add_argument('--overwrite', action='store_true', help='Flag for overwriting the result outputs')

    # Parameters for the dataloader
    parser.add_argument('--ttt_sequence_length', default=[3], type=lambda x: list(map(int, x.split(','))),
                        help='Number(s) of frames used in frame sequences used for TTT. It can be one number or '
                             'comma-separated e.g. 3,4,5')
    parser.add_argument('--ttt_resolution', default=480, type=int,
                        help='Frame resolution of the frames in the frame sequence')
    parser.add_argument('--ttt_batch_size', default=1, type=int,
                        help='Number of sequences used for each batch for TTT')

    # Parameters for the loss
    parser.add_argument('--ttt_loss', default="tt-mcc", type=str, choices=['tt-mcc', 'tt-ae', 'tt-ent'],
                        help='Loss function used')  # CHANGING
    parser.add_argument('--ttt_frozen_layers', default=None, type=str,
                        help='Parts of the network that remain frozen during training')

    parser.add_argument('--ttt_max_jump_step', default=[10], type=lambda x: list(map(int, x.split(','))),
                        help='Maximum jump step between two frames in the frame sequence. It can be one number or '
                             'comma-separated e.g. 5,10,15')
    parser.add_argument('--ttt_number_jump_steps', default=10, type=int,
                        help='Number of different jump step sampled for a video. For each jump step sampled')
    parser.add_argument('--ttt_number_iterations_per_jump_step', default=10, type=int,
                        help='Number of iterations run for one --ttt_jump_steps')

    # Parameters for augmentations
    parser.add_argument('--ttt_augmentations', default='none',
                        choices=['none', 'colour', 'geometric', 'geometric,colour', 'colour,geometric'],
                        help='Type of augmentations used on the frames of the training sequences')
    parser.add_argument('--ttt_scale', default=[1., 1.], type=lambda x: list(map(float, x.split(','))),
                        help='Range of scale used for the frame augmentation')
    parser.add_argument('--ttt_ratio', default=[1., 1.], type=lambda x: list(map(float, x.split(','))),
                        help='Range of ratio used for the frame augmentation')

    # Parameters of the original STCN method
    parser.add_argument('--top', type=int, default=20,
                        help='Top-k agencies used for inference')
    parser.add_argument('--mem_every', default=5, type=int,
                        help='Interval for adding frames in the memory bank')
    parser.add_argument('--amp', action='store_true',
                        help='Flag for mixed precision processing')
    parser.add_argument('--include_last', action='store_true',
                        help='include last frame as temporary memory?')

    args = parser.parse_args()
    return args


def save_config(config_filename, args):
    model_parameters = {
        "MODEL_FILENAME": args.model_filename,
        "TOP": args.top,
        "MEM_EVERY": args.mem_every,
        "INCLUDE_LAST": args.include_last,
        "AMP": args.amp,
    }

    train_parameters = {
        "TTT_LR": args.ttt_lr,
        "TTT_BATCH_SIZE": args.ttt_batch_size,
    }

    loss_parameters = {
        "TTT_LOSS": args.ttt_loss,
        "TTT_FROZEN_LAYERS": args.ttt_frozen_layers,
        "TTT_NUMBER_JUMP_STEPS": args.ttt_number_jump_steps,
        "TTT_NUMBER_ITERATIONS_PER_JUMP_STEP": args.ttt_number_iterations_per_jump_step,
    }

    sampling_parameters = {
        "TTT_RESOLUTION": args.ttt_resolution,
        "TTT_SEQUENCE_LENGTH": args.ttt_sequence_length,
        "TTT_MAX_JUMP_STEP": args.ttt_max_jump_step,
    }

    augmentation_parameters = {
        "TTT_AUGMENTATIONS": args.ttt_augmentations,
        "TTT_SCALE": args.ttt_scale,
        "TTT_RATIO": args.ttt_ratio,
    }

    global_parameters = {
        "CONFIG_NAME": args.config_name,
        "OVERWRITE": args.overwrite,
    }

    parameters = {
        "STCN_MODEL": model_parameters,
        "TEST_TIME_TRAIN": train_parameters,
        "TEST_TIME_LOSS": loss_parameters,
        "SEQUENCE_SAMPLING": sampling_parameters,
        "AUGMENTATIONS": augmentation_parameters,
        "GLOBAL": global_parameters,
    }

    with open(config_filename, 'w') as file:
        yaml.dump(parameters, file)


if __name__ == "__main__":

    args = parse_arguments()
    config_filename = os.path.join("configs", f'{args.config_name}.yaml')
    os.makedirs(os.path.dirname(config_filename), exist_ok=True)
    save_config(config_filename, args)
    print(f"saved config {config_filename}")



