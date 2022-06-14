"""
Run file to train an autoencoder.
"""

import argparse
import torch.utils.data as data
import pytorch_lightning as pl

import sys
sys.path.append('../')
from models.ae import Autoencoder
from experiments.datasets import Causal3DDataset, InterventionalPongDataset, VoronoiDataset, PinballDataset, BallInBoxesDataset
from experiments.utils import train_model, print_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--cluster', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--c_hid', type=int, default=64)
    parser.add_argument('--num_latents', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--noise_level', type=float, default=0.05)
    parser.add_argument('--regularizer_weight', type=float, default=1e-6)
    parser.add_argument('--logger_name', type=str, default='')
    parser.add_argument('--files_to_save', type=str, nargs='+', default='')

    args = parser.parse_args()
    pl.seed_everything(args.seed)

    print('Loading datasets...')
    if 'ball_in_boxes' in args.data_dir:
        DataClass = BallInBoxesDataset
    elif 'pong' in args.data_dir:
        DataClass = InterventionalPongDataset
    elif 'causal3d' in args.data_dir:
        DataClass = Causal3DDataset
    elif 'voronoi' in args.data_dir:
        DataClass = VoronoiDataset
    elif 'pinball' in args.data_dir:
        DataClass = PinballDataset
    else:
        DataClass = Causal3DDataset
    
    train_dataset = DataClass(
        data_folder=args.data_dir, split='train', single_image=True, seq_len=1)
    val_dataset = DataClass(
        data_folder=args.data_dir, split='val', single_image=True, seq_len=1)
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=args.num_workers)
    print(f'Length training dataset: {len(train_dataset)} / Train loader: {len(train_loader)}')
    print(f'Length val dataset: {len(val_dataset)} / Test loader: {len(val_loader)}')

    args.max_iters = args.max_epochs * len(train_loader)
    model_args = vars(args)
    model_args['img_width'] = train_dataset.get_img_width()
    if hasattr(train_dataset, 'get_inp_channels'):
        model_args['c_in'] = train_dataset.get_inp_channels()
    print(f'Image size: {model_args["img_width"]}')
    model_class = Autoencoder
    logger_name = f'AE_{args.num_latents}l_{args.c_hid}hid'
    args_logger_name = model_args.pop('logger_name')
    if len(args_logger_name) > 0:
        logger_name += '/' + args_logger_name

    print_params(logger_name, model_args)

    train_model(model_class=model_class,
                train_loader=train_loader,
                val_loader=val_loader,
                progress_bar_refresh_rate=0 if args.cluster else 1,
                logger_name=logger_name,
                check_val_every_n_epoch=min(10, args.max_epochs),
                gradient_clip_val=0.1,
                **model_args)
