"""
Run file for training CITRIS-NF on a pretrained autoencoder architecture.
"""

import argparse
import os
import torch
import torch.utils.data as data

import sys
sys.path.append('../')
from models.citris_nf import CITRISNF
from models.icitris_nf import iCITRISNF
from experiments.utils import train_model, load_datasets, get_default_parser, print_params


def encode_dataset(model, datasets):
    if isinstance(datasets, data.Dataset):
        datasets = [datasets]
    if any([isinstance(d, dict) for d in datasets]):
        new_datasets = []
        for d in datasets:
            if isinstance(d, dict):
                new_datasets += list(d.values())
            else:
                new_datasets.append(d)
        datasets = new_datasets
    for dataset in datasets:
        autoencoder_folder = model.hparams.autoencoder_checkpoint.rsplit('/', 1)[0]
        encoding_folder = os.path.join(autoencoder_folder, 'encodings/')
        os.makedirs(encoding_folder, exist_ok=True)
        encoding_filename = os.path.join(encoding_folder, f'{model.hparams.data_folder}_{dataset.split_name}.pt')
        if not os.path.exists(encoding_filename):
            encodings = dataset.encode_dataset(model.autoencoder.encoder)
            torch.save(encodings, encoding_filename)
        else:
            dataset.load_encodings(encoding_filename)


if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument('--autoencoder_checkpoint', type=str,
                        required=True)
    parser.add_argument('--c_hid', type=int, default=64)
    parser.add_argument('--num_flows', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=8)
    parser.add_argument('--flow_act_fn', type=str, default='silu')
    parser.add_argument('--hidden_per_var', type=int, default=16)
    parser.add_argument('--act_fn', type=str, default='silu')
    parser.add_argument('--num_latents', type=int, default=32)
    parser.add_argument('--classifier_lr', type=float, default=4e-3)
    parser.add_argument('--classifier_momentum', type=float, default=0.0)
    parser.add_argument('--classifier_gumbel_temperature', type=float, default=1.0)
    parser.add_argument('--classifier_use_normalization', action='store_true')
    parser.add_argument('--classifier_use_conditional_targets', action='store_true')
    parser.add_argument('--beta_t1', type=float, default=1.0)
    parser.add_argument('--beta_classifier', type=float, default=2.0)
    parser.add_argument('--beta_mi_estimator', type=float, default=2.0)
    parser.add_argument('--lambda_reg', type=float, default=0.0)
    parser.add_argument('--num_graph_samples', type=int, default=8)
    parser.add_argument('--autoregressive_prior', action='store_true')
    parser.add_argument('--model', type=str, default='iCITRISNF')
    parser.add_argument('--cluster_log_folder', action="store_true")
    parser.add_argument('--lambda_sparse', type=float, default=0.1)
    parser.add_argument('--mi_estimator_comparisons', type=int, default=1)
    parser.add_argument('--graph_learning_method', type=str, default="ENCO")
    parser.add_argument('--enco_postprocessing', action="store_true")
    parser.add_argument('--use_notears_regularizer', action="store_true")

    args = parser.parse_args()
    model_args = vars(args)

    datasets, data_loaders, data_name = load_datasets(args)

    model_args['data_folder'] = [s for s in args.data_dir.split('/') if len(s) > 0][-1]
    model_args['width'] = datasets['train'].get_img_width()
    model_args['num_causal_vars'] = datasets['train'].num_vars()
    model_args['max_iters'] = args.max_epochs * len(data_loaders['train'])
    if args.model == 'CITRISNF':
        model_class = CITRISNF
    elif args.model == 'iCITRISNF':
        model_class = iCITRISNF
    else:
        assert False, f'Unknown model class \"{args.model}\"'
    if not args.cluster_log_folder:
        logger_name = f'{args.model}_{args.num_latents}l_{args.num_causal_vars}b_{args.c_hid}hid_{data_name}'
    else:
        logger_name = 'Cluster'
    args_logger_name = model_args.pop('logger_name')
    if len(args_logger_name) > 0:
        logger_name += '/' + args_logger_name

    print_params(logger_name, model_args)
    
    check_val_every_n_epoch = model_args.pop('check_val_every_n_epoch')
    if check_val_every_n_epoch <= 0:
        check_val_every_n_epoch = 5 if not args.cluster else 10
    train_model(model_class=model_class,
                train_loader=data_loaders['train'],
                val_loader=data_loaders['val_triplet'],
                test_loader=data_loaders['test_triplet'],
                logger_name=logger_name,
                check_val_every_n_epoch=check_val_every_n_epoch,
                progress_bar_refresh_rate=0 if args.cluster else 1,
                callback_kwargs={'dataset': datasets['train'], 
                                 'correlation_dataset': datasets['val'],
                                 'correlation_test_dataset': datasets['test'],
                                 'add_enco_sparsification': args.enco_postprocessing},
                var_names=datasets['train'].target_names(),
                op_before_running=lambda model: encode_dataset(model, list(datasets.values())),
                save_last_model=True,
                cluster_logging=args.cluster,
                val_track_metric='val_comb_loss',
                **model_args)
