"""
Run file for training a supervised CNN on predicting the causal factors from images.
This CNN is used for the triplet evaluation.
"""

import argparse
import os
from glob import glob
from tqdm.auto import tqdm
import torch
import torch.utils.data as data

import sys
sys.path.append('../')
from models.shared import CausalEncoder
from models.ae import Autoencoder
from experiments.datasets import Causal3DDataset, InterventionalPongDataset, VoronoiDataset, PinballDataset, BallInBoxesDataset
from experiments.utils import train_model, print_params


@torch.no_grad()
def encode_datasets(datasets, autoencoder_checkpoint, cluster=False):
    if isinstance(datasets, data.Dataset):
        datasets = [datasets]
    if os.path.isdir(autoencoder_checkpoint):
        autoencoder_checkpoint = sorted(glob(os.path.join(autoencoder_checkpoint, '*.ckpt')))[0]
    ae = Autoencoder.load_from_checkpoint(autoencoder_checkpoint)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    ae.eval()
    ae.to(device)
    for dataset in datasets:
        loader = data.DataLoader(dataset, batch_size=512, shuffle=False, drop_last=False, pin_memory=True)
        recons = []
        if not cluster:
            loader = tqdm(loader, leave=False, desc='Encoding dataset')
        for batch in loader:
            imgs, *_ = batch
            out = ae.forward(imgs.to(device))
            recons.append(out.detach().cpu())
        recons = torch.cat(recons, dim=0)
        dataset.imgs = recons
        dataset.encodings_active = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster', action="store_true")
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--data_img_width', type=int, default=64)
    parser.add_argument('--c_hid', type=int, default=64)
    parser.add_argument('--act_fn', type=str, default='silu')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--angle_reg_weight', type=float, default=0.1)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=10)
    parser.add_argument('--autoencoder_checkpoint', type=str, default=None)
    parser.add_argument('--logger_name', type=str, default='')
    parser.add_argument('--files_to_save', type=str, nargs='+', default='')
    parser.add_argument('--cluster_log_folder', action="store_true")

    args, _ = parser.parse_known_args()

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
        data_folder=args.data_dir, split='train', single_image=True, return_latents=True, 
        coarse_vars=False, img_width=args.data_img_width)
    val_dataset = DataClass(
        data_folder=args.data_dir, split='val', single_image=True, return_latents=True,
        causal_vars=train_dataset.target_names(), coarse_vars=False, img_width=args.data_img_width)

    if args.autoencoder_checkpoint is not None:
        encode_datasets([train_dataset, val_dataset], args.autoencoder_checkpoint, cluster=args.cluster)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                   shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size,
                                  shuffle=False, drop_last=False, num_workers=args.num_workers)

    args.max_iters = args.max_epochs * len(train_loader)
    model_args = vars(args)
    model_args['img_width'] = train_dataset.get_img_width()
    model_class = CausalEncoder
    data_name = args.data_dir.split("_",1)[-1].replace("/","")
    if isinstance(train_dataset, InterventionalPongDataset):
        data_name = 'pong_' + data_name
    elif isinstance(train_dataset, PinballDataset):
        data_name = 'pinball_' + data_name
    if not args.cluster_log_folder:
        logger_name = f'CausalEncoder_{data_name}_{args.data_img_width}width_{args.c_hid}hid'
    else:
        logger_name = f'Cluster'
    args_logger_name = model_args.pop('logger_name')
    if len(args_logger_name) > 0:
        logger_name += '/' + args_logger_name
    
    print_params(logger_name, model_args)

    check_val_every_n_epoch = model_args.pop('check_val_every_n_epoch')
    train_model(model_class=model_class,
                train_loader=train_loader,
                val_loader=val_loader,
                progress_bar_refresh_rate=0 if args.cluster else 1,
                logger_name=logger_name,
                causal_var_info=train_dataset.get_causal_var_info(),
                c_in=train_dataset.get_inp_channels(),
                check_val_every_n_epoch=check_val_every_n_epoch,
                gradient_clip_val=1.0,
                **model_args)
