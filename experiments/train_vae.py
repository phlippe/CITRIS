"""
Run file to train iCITRIS-VAE, CITRIS-VAE, iVAE*, and SlowVAE.
"""

import argparse
import torch.utils.data as data

import sys
sys.path.append('../')
from models.icitris_vae import iCITRISVAE
from models.citris_vae import CITRISVAE
from models.baselines import iVAE, SlowVAE
from experiments.utils import train_model, load_datasets, get_default_parser, print_params


if __name__ == '__main__':
    parser = get_default_parser()
    parser.add_argument('--model', type=str, default='iCITRISVAE')
    parser.add_argument('--c_hid', type=int, default=32)
    parser.add_argument('--decoder_num_blocks', type=int, default=1)
    parser.add_argument('--act_fn', type=str, default='silu')
    parser.add_argument('--num_latents', type=int, default=16)
    parser.add_argument('--classifier_lr', type=float, default=4e-3)
    parser.add_argument('--classifier_momentum', type=float, default=0.0)
    parser.add_argument('--classifier_gumbel_temperature', type=float, default=1.0)
    parser.add_argument('--classifier_use_normalization', action='store_true')
    parser.add_argument('--classifier_use_conditional_targets', action='store_true')
    parser.add_argument('--kld_warmup', type=int, default=0)
    parser.add_argument('--beta_t1', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lambda_reg', type=float, default=0.0)
    parser.add_argument('--autoregressive_prior', action='store_true')
    parser.add_argument('--beta_classifier', type=float, default=2.0)
    parser.add_argument('--beta_mi_estimator', type=float, default=2.0)
    parser.add_argument('--lambda_sparse', type=float, default=0.02)
    parser.add_argument('--mi_estimator_comparisons', type=int, default=1)
    parser.add_argument('--graph_learning_method', type=str, default="ENCO")

    args = parser.parse_args()
    model_args = vars(args)

    datasets, data_loaders, data_name = load_datasets(args)

    model_args['data_folder'] = [s for s in args.data_dir.split('/') if len(s) > 0][-1]
    model_args['img_width'] = datasets['train'].get_img_width()
    model_args['num_causal_vars'] = datasets['train'].num_vars()
    model_args['max_iters'] = args.max_epochs * len(data_loaders['train'])
    if hasattr(datasets['train'], 'get_inp_channels'):
        model_args['c_in'] = datasets['train'].get_inp_channels()
    model_name = model_args.pop('model')
    if model_name == 'iCITRISVAE':
        model_class = iCITRISVAE
    elif model_name == 'CITRISVAE':
        model_class = CITRISVAE
    elif model_name == 'iVAE':
        model_class = iVAE
    elif model_name == 'SlowVAE':
        model_class = SlowVAE
    logger_name = f'{model_name}_{args.num_latents}l_{model_args["num_causal_vars"]}b_{args.c_hid}hid_{data_name}'
    args_logger_name = model_args.pop('logger_name')
    if len(args_logger_name) > 0:
        logger_name += '/' + args_logger_name

    print_params(logger_name, model_args)
    
    check_val_every_n_epoch = model_args.pop('check_val_every_n_epoch')
    if check_val_every_n_epoch <= 0:
        check_val_every_n_epoch = 2 if not args.cluster else 25
    train_model(model_class=model_class,
                train_loader=data_loaders['train'],
                val_loader=data_loaders['val_triplet'],
                test_loader=data_loaders['test_triplet'],
                logger_name=logger_name,
                check_val_every_n_epoch=check_val_every_n_epoch,
                progress_bar_refresh_rate=0 if args.cluster else 1,
                callback_kwargs={'dataset': datasets['train'], 
                                 'correlation_dataset': datasets['val'],
                                 'correlation_test_dataset': datasets['test']},
                var_names=datasets['train'].target_names(),
                save_last_model=True,
                cluster_logging=args.cluster,
                **model_args)
