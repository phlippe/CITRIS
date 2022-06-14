import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import numpy as np
from collections import defaultdict

import sys
sys.path.append('../')
from models.shared import CosineWarmupScheduler, SineWarmupScheduler, get_act_fn, kl_divergence, log_dict, Encoder, Decoder
from models.shared import CausalEncoder
from models.baselines.ivae import iVAE


class SlowVAE(iVAE):
    """ Module of the SlowVAE, heavily inspired from https://github.com/bethgelab/slow_disentanglement """

    def __init__(self, *args, 
                       gamma=1.0,
                       **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_net = None

    def _get_loss(self, batch, mode='train'):
        if len(batch) == 2:
            imgs, target = batch
            labels = imgs
        else:
            imgs, labels, target = batch
        z_mean, z_logstd = self.encoder(imgs.flatten(0, 1))
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        x_rec = self.decoder(z_sample.unflatten(0, imgs.shape[:2]).flatten(0, 1))
        z_sample, z_mean, z_logstd, x_rec = [t.unflatten(0, (imgs.shape[0], -1)) for t in [z_sample, z_mean, z_logstd, x_rec]]

        base_kld = kl_divergence(z_mean, z_logstd).sum(dim=-1)
        base_kld = base_kld[:,1:] + base_kld[:,:-1]

        # Taken from SlowVAE implementation
        normal_entropy = 0.5 * (2 * z_logstd + np.log(2 * np.pi * np.e))
        normal_entropy = 0.5 * (normal_entropy[:,1:] + normal_entropy[:,:-1])  # Averaging elem 1 and 2
        laplace_kld = self.compute_cross_ent_laplace(z_mean[:,:-1] - z_mean[:,1:], z_logstd[:,:-1], 1) + \
                      self.compute_cross_ent_laplace(z_mean[:,1:] - z_mean[:,:-1], z_logstd[:,1:], 1)
        time_kld = (laplace_kld - normal_entropy).sum(dim=-1)

        rec_loss = F.mse_loss(x_rec, labels, reduction='none').sum(dim=[-3, -2, -1])
        rec_loss = rec_loss[:,1:] + rec_loss[:,:-1]

        kld_factor = self.kld_scheduler.get_factor(self.global_step)
        loss = (kld_factor * (base_kld * self.hparams.beta_t1 + time_kld * self.hparams.gamma) + rec_loss).mean()
        loss = loss / (imgs.shape[1] - 1)

        self.log(f'{mode}_base_kld', base_kld.mean())
        self.log(f'{mode}_time_kld', time_kld.mean())
        self.log(f'{mode}_rec_loss', rec_loss.mean())
        if mode == 'train':
            self.log(f'{mode}_kld_scheduling', kld_factor)

        return loss

    def compute_cross_ent_laplace(self, mean, logvar, rate_prior):
        normal_dist = torch.distributions.normal.Normal(
            torch.zeros(self.hparams.num_latents, device=self.device),
            torch.ones(self.hparams.num_latents, device=self.device))
        var = torch.exp(logvar)
        sigma = torch.sqrt(var)
        ce = - np.log(rate_prior / 2) + rate_prior * sigma *\
             np.sqrt(2 / np.pi) * torch.exp(- mean**2 / (2 * var)) -\
             rate_prior * mean * (
                     1 - 2 * normal_dist.cdf(mean / sigma))
        return ce