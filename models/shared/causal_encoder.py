import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import numpy as np
from collections import OrderedDict, defaultdict

import sys
sys.path.append('../../')
from models.shared.modules import CosineWarmupScheduler, TanhScaled
from models.shared.encoder_decoder import Encoder
from models.shared.utils import get_act_fn


class CausalEncoder(pl.LightningModule):
    """ Network trained supervisedly on predicted the ground truth causal factors from input data, e.g. images """

    def __init__(self, c_hid, lr, 
                       causal_var_info,
                       warmup=500, max_iters=100000,
                       img_width=64,
                       weight_decay=0.0,
                       act_fn='silu',
                       single_linear=False,
                       c_in=-1,
                       angle_reg_weight=0.1,
                       **kwargs):
        """
        Parameters
        ----------
        c_hid : int
                Hidden dimensionality to use in the network
        lr : float
             Learning rate to use for training.
        causal_var_info : OrderedDict
                          Dictionary with causal variable name to prediction
                          information.
        warmup : int
                 Number of learning rate warmup steps
        max_iters : int
                    Number of max. training iterations. Needed for 
                    cosine annealing of the learning rate.
        img_width : int
                    Width of the input image (assumed to be equal to height)
        weight_decay : float
                       Weight decay to apply during training.
        act_fn : str
                 Activation function to use in the network.
        single_linear : bool
                        If True, a linear network is used.
        c_in : int
               Number of input channels (3 for RGB)
        angle_reg_weight : float
                           Regularization weight for angle predictions.
        """
        super().__init__()
        self.save_hyperparameters()

        # Base Network
        if not self.hparams.single_linear:
            act_fn_func = get_act_fn(self.hparams.act_fn)
            self.encoder = Encoder(num_latents=self.hparams.c_hid,
                                   c_in=max(3, self.hparams.c_in),
                                   c_hid=self.hparams.c_hid,
                                   width=self.hparams.img_width,
                                   act_fn=act_fn_func,
                                   variational=False)
        else:
            self.encoder = nn.Sequential(
                    nn.Linear(self.hparams.c_in, self.hparams.c_hid),
                    nn.Tanh(),
                    nn.Linear(self.hparams.c_hid, self.hparams.c_hid),
                    nn.Tanh()
                )

        # For each causal variable, we create a separate layer as 'head'.
        # Depending on the domain, we use different specifications for the head.
        self.pred_layers = nn.ModuleDict()
        for var_key in self.hparams.causal_var_info:
            var_info = self.hparams.causal_var_info[var_key]
            if var_info.startswith('continuous'):  # Regression
                self.pred_layers[var_key] = nn.Sequential(
                    nn.Linear(self.hparams.c_hid, 1),
                    TanhScaled(scale=float(var_info.split('_')[-1]))
                )
            elif var_info.startswith('angle'):  # Predicting 2D vector for the angle
                self.pred_layers[var_key] = nn.Linear(self.hparams.c_hid, 2)
            elif var_info.startswith('categ'):  # Classification
                self.pred_layers[var_key] = nn.Linear(self.hparams.c_hid, int(var_info.split('_')[-1]))
            else:
                assert False, f'Do not know how to handle key \"{var_key}\" in CausalEncoder initialization.'
        self.val_pred_list = []
        self.all_val_dists = []

    def forward(self, x):
        z = self.encoder(x)
        v = OrderedDict()
        for var_key in self.hparams.causal_var_info:
            v[var_key] = self.pred_layers[var_key](z)
        return v

    def predict_causal_vars(self, x):
        z = self.encoder(x)
        vals = []
        for var_key in self.hparams.causal_var_info:
            var_info = self.hparams.causal_var_info[var_key]
            pred = self.pred_layers[var_key](z)
            if var_info.startswith('continuous'):
                pass  # Nothing special to do
            elif var_info.startswith('angle'):
                pred = torch.atan2(pred[...,0], pred[...,1])[...,None]
                pred = torch.where(pred < 0, pred + 2*np.pi, pred)
            elif var_info.startswith('categ'):
                pred = pred.argmax(dim=-1)[...,None]
            vals.append(pred)
        vals = torch.cat(vals, dim=-1)
        return vals

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 
                                lr=self.hparams.lr, 
                                weight_decay=self.hparams.weight_decay)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def calculate_loss_distance(self, pred_dict, gt_vec, keep_sign=False):
        # Function for calculating the loss and distance between predictions (pred_dict) and
        # ground truth (gt_vec) for every causal variable in the dictionary.
        losses = OrderedDict()
        dists = OrderedDict()
        norm_dists = OrderedDict()
        for i, var_key in enumerate(pred_dict):
            var_info = self.hparams.causal_var_info[var_key]
            gt_val = gt_vec[...,i]
            if var_info.startswith('continuous'):
                # MSE loss
                losses[var_key] = F.mse_loss(pred_dict[var_key].squeeze(dim=-1),
                                             gt_val, reduction='none')
                dists[var_key] = (pred_dict[var_key].squeeze(dim=-1) - gt_val)
                if not keep_sign:
                    dists[var_key] = dists[var_key].abs()
                norm_dists[var_key] = dists[var_key] / float(var_info.split('_')[-1])
            elif var_info.startswith('angle'):
                # Cosine similarity loss
                vec = torch.stack([torch.sin(gt_val), torch.cos(gt_val)], dim=-1)
                cos_sim = F.cosine_similarity(pred_dict[var_key], vec, dim=-1)
                losses[var_key] = 1 - cos_sim
                if self.training:
                    norm = pred_dict[var_key].norm(dim=-1, p=2.0)
                    losses[var_key + '_reg'] = self.hparams.angle_reg_weight * (2 - norm) ** 2
                dists[var_key] = torch.where(cos_sim > (1-1e-7), torch.zeros_like(cos_sim), torch.acos(cos_sim.clamp_(min=-1+1e-7, max=1-1e-7)))
                dists[var_key] = dists[var_key] / np.pi * 180.0  # rad to degrees
                norm_dists[var_key] = dists[var_key] / 180.0
            elif var_info.startswith('categ'):
                # Cross entropy loss
                gt_val = gt_val.long()
                pred = pred_dict[var_key]
                if len(pred.shape) > 2:
                    pred = pred.flatten(0, -2)
                    gt_val = gt_val.flatten(0, -1)
                losses[var_key] = F.cross_entropy(pred, gt_val, reduction='none')
                if len(pred_dict[var_key]) > 2:
                    losses[var_key] = losses[var_key].reshape(pred_dict[var_key].shape[:-1])
                    gt_val = gt_val.reshape(pred_dict[var_key].shape[:-1])
                dists[var_key] = (gt_val != pred_dict[var_key].argmax(dim=-1)).float()
                norm_dists[var_key] = dists[var_key]
            else:
                assert False, f'Do not know how to handle key \"{var_key}\" in calculating distances and losses.'
        for var_key in losses:
            losses[var_key] = losses[var_key].mean()
        return losses, dists, norm_dists

    def get_distances(self, imgs, labels, return_norm_dists=False, return_v_dict=False):
        # Returns the distances between causal factors predicted from an image, and the labels
        v_dict = self.forward(imgs)
        losses, dists, norm_dists = self.calculate_loss_distance(v_dict, labels)
        if not self.training:
            self.val_pred_list.append(v_dict)
            self.all_val_dists.append(dists)
        returns = [losses, dists]
        if return_norm_dists:
            returns.append(norm_dists)
        if return_v_dict:
            keys = list(v_dict.keys())
            for i, var_key in enumerate(keys):
                v_dict[var_key + '_gt'] = labels[...,i]
            returns.append(v_dict)
        return returns

    def _get_loss(self, batch, mode='train'):
        imgs, *_, labels = batch
        losses, dists = self.get_distances(imgs, labels)
        loss = sum([losses[key] for key in losses])

        if mode is not None:
            for var_key in losses:
                self.log(f'{mode}_{var_key}_loss', losses[var_key])
            for var_key in dists:
                self.log(f'{mode}_{var_key}_dist', dists[var_key].mean())

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='train')
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='val')
        self.log('val_loss', loss)

    def validation_epoch_end(self, *args, **kwargs):
        super().validation_epoch_end(*args, **kwargs)
        if len(self.all_val_dists) > 0:
            if self.current_epoch > 0:
                for key in self.all_val_dists[0]:
                    vals = torch.cat([d[key] for d in self.all_val_dists], dim=0)
                    self.logger.experiment.add_histogram(key, vals, self.current_epoch)
        self.all_val_dists = []
        self.val_pred_list = []

    @staticmethod
    def get_callbacks(*args, **kwargs):
        hist_callback = CausalEncoderHistogramCallback()
        lr_callback = LearningRateMonitor('step')
        return [hist_callback, lr_callback]


class CausalEncoderHistogramCallback(pl.Callback):
    """ Visualize the predicted values in a histogram """

    def __init__(self):
        super().__init__()
        self.prefix = 'causal_encoder'

    def on_validation_epoch_end(self, trainer, pl_module):
        def log_fig(tag, fig):
            trainer.logger.experiment.add_figure(f'{self.prefix}{tag}', fig, global_step=trainer.global_step)
            plt.close(fig)

        pred_dict = OrderedDict()
        pred_list = pl_module.val_pred_list
        if len(pred_list) == 0:
            return

        for key in pred_list[0]:
            if key.endswith('_gt'):
                continue
            pred_dict[key] = torch.cat([d[key] for d in pred_list], dim=0).cpu().numpy()

        for key in pred_dict:
            fig = plt.figure(figsize=(4,4))
            plt.title(f'Histogram {key}')
            vals = pred_dict[key]
            if vals.shape[-1] == 2:
                plt.hist2d(x=vals[...,0], y=vals[...,1], bins=25)
            elif vals.shape[-1] == 1 or len(vals.shape) == 1:
                plt.hist(x=vals, bins=10)
            else:
                num_categs = vals.shape[-1]
                vals = vals.argmax(axis=-1)
                plt.hist(vals, np.arange(-1.0, num_categs + 1.0, 1.0))
            log_fig(f'_{key}', fig)

        pl_module.val_pred_list = []