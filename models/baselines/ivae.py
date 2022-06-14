"""
Model architectures for iVAE* and SlowVAE
"""

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
from models.shared import CosineWarmupScheduler, SineWarmupScheduler, get_act_fn, kl_divergence, gaussian_log_prob, log_dict, Encoder, Decoder, SimpleEncoder, SimpleDecoder
from models.shared import ImageLogCallback, CausalEncoder, AutoregLinear
from models.ae import Autoencoder
from models.baselines.utils import BaselineCorrelationMetricsLogCallback
from models.shared import AutoregNormalizingFlow


class iVAE(pl.LightningModule):
    """ 
    Module for implementing the adapted iVAE 
    It is similarly structured as CITRIS-VAE, although being reduced
    to a standard VAE instead of with a full transition prior etc.
    """

    def __init__(self, c_hid, num_latents, lr, 
                       num_causal_vars,
                       c_in=3,
                       warmup=100, 
                       max_iters=100000,
                       kld_warmup=100,
                       beta_t1=1.0,
                       img_width=32,
                       decoder_num_blocks=1,
                       act_fn='silu',
                       causal_var_info=None,
                       causal_encoder_checkpoint=None,
                       use_flow_prior=True,
                       autoencoder_checkpoint=None,
                       autoregressive_prior=False,
                       **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        act_fn_func = get_act_fn(self.hparams.act_fn)
        if autoencoder_checkpoint is None:
            if self.hparams.img_width == 32:
                self.encoder = SimpleEncoder(num_input_channels=self.hparams.c_in,
                                             base_channel_size=self.hparams.c_hid,
                                             latent_dim=self.hparams.num_latents)
                self.decoder = SimpleDecoder(num_input_channels=self.hparams.c_in,
                                             base_channel_size=self.hparams.c_hid,
                                             latent_dim=self.hparams.num_latents)
            else:
                self.encoder = Encoder(num_latents=self.hparams.num_latents,
                                          c_hid=self.hparams.c_hid,
                                          c_in=self.hparams.c_in,
                                          width=self.hparams.img_width,
                                          act_fn=act_fn_func,
                                          use_batch_norm=False)
                self.decoder = Decoder(num_latents=self.hparams.num_latents,
                                          c_hid=self.hparams.c_hid,
                                          c_out=self.hparams.c_in,
                                          width=self.hparams.img_width,
                                          num_blocks=self.hparams.decoder_num_blocks,
                                          act_fn=act_fn_func,
                                          use_batch_norm=False)
        else:
            self.autoencoder = Autoencoder.load_from_checkpoint(self.hparams.autoencoder_checkpoint)
            for p in self.autoencoder.parameters():
                p.requires_grad_(False)
            def encod_function(inp):
                mean = self.autoencoder.encoder(inp)
                log_std = torch.zeros_like(mean).fill_(np.log(self.autoencoder.hparams.noise_level))
                return mean, log_std
            self.encoder = encod_function
            self.decoder = lambda inp: self.autoencoder.decoder(inp)
        # Prior of p(z^t+1|z^t,I^t)
        if self.hparams.autoregressive_prior:
            self.prior_net_cond = nn.Linear(num_latents + num_causal_vars, 16 * num_latents)
            self.prior_net_init = AutoregLinear(num_latents, 1, 16, diagonal=False)
            self.prior_net_head = nn.Sequential(
                    nn.SiLU(),
                    AutoregLinear(num_latents, 16, 16, diagonal=True),
                    nn.SiLU(),
                    AutoregLinear(num_latents, 16, 2, diagonal=True)
                )
        else:
            self.prior_net = nn.Sequential(
                    nn.Linear(num_latents + num_causal_vars, self.hparams.c_hid*8),
                    nn.SiLU(),
                    nn.Linear(self.hparams.c_hid*8, self.hparams.c_hid*8),
                    nn.SiLU(),
                    nn.Linear(self.hparams.c_hid*8, num_latents*2)
                )
        if self.hparams.use_flow_prior:
            self.flow = AutoregNormalizingFlow(self.hparams.num_latents,
                                               num_flows=4,
                                               act_fn=nn.SiLU,
                                               hidden_per_var=16)
        self.kld_scheduler = SineWarmupScheduler(kld_warmup, start_factor=0.01)
        # Causal Encoder loading
        if self.hparams.causal_encoder_checkpoint is not None:
            self.causal_encoder_true_epoch = int(1e5)  # We want to log the true causal encoder distance once
            self.causal_encoder = CausalEncoder.load_from_checkpoint(self.hparams.causal_encoder_checkpoint)
            for p in self.causal_encoder.parameters():
                p.requires_grad_(False)
        else:
            self.causal_encoder = None
        self.target_assignment = None
        self.output_to_input = None
        self.register_buffer('last_target_assignment', torch.zeros(num_latents, num_causal_vars))
        self.all_val_dists = defaultdict(list)

    def forward(self, x):
        z_mean, z_logstd = self.encoder(x)
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        x_rec = self.decoder(z_sample)
        return x_rec, z_sample, z_mean, z_logstd

    def encode(self, x, random=True):
        z_mean, z_logstd = self.encoder(x)
        if random:
            z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        else:
            z_sample = z_mean
        if self.hparams.use_flow_prior:
            z_sample, _ = self.flow(z_sample)
        return z_sample

    def _run_prior(self, z_sample, target):
        inp = torch.cat([z_sample[:,:-1], target], dim=-1)
        true_out = z_sample[:,1:]
        inp = inp.flatten(0, 1)
        true_out = true_out.flatten(0, 1)
        if self.hparams.autoregressive_prior:
            cond_feats = self.prior_net_cond(inp)
            init_feats = self.prior_net_init(true_out)
            comb_feats = cond_feats + init_feats
            out_feats = self.prior_net_head(comb_feats)
            out_feats = out_feats.unflatten(-1, (-1, 2))
            prior_mean, prior_logstd = out_feats.unbind(dim=-1)
        else:
            prior_mean, prior_logstd = self.prior_net(inp).chunk(2, dim=-1)
        prior_mean = prior_mean.unflatten(0, target.shape[:2])
        prior_logstd = prior_logstd.unflatten(0, target.shape[:2])
        return prior_mean, prior_logstd

    def sample_latent(self, batch_size, for_intervention=False):
        sample = torch.randn(batch_size, self.hparams.num_latents, device=self.device)
        return sample

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _get_loss(self, batch, mode='train'):
        if len(batch) == 2:
            imgs, target = batch
            labels = imgs
        else:
            imgs, labels, target = batch
        z_mean, z_logstd = self.encoder(imgs.flatten(0, 1))
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        x_rec = self.decoder(z_sample.unflatten(0, imgs.shape[:2])[:,1:].flatten(0, 1))
        z_sample, z_mean, z_logstd, x_rec = [t.unflatten(0, (imgs.shape[0], -1)) for t in [z_sample, z_mean, z_logstd, x_rec]]

        if not self.hparams.use_flow_prior:
            # prior_t1_mean, prior_t1_logstd = self.prior_net(torch.cat([z_sample[:,:-1], target], dim=-1)).chunk(2, dim=-1)
            prior_t1_mean, prior_t1_logstd = self._run_prior(z_sample, target)
            kld = kl_divergence(z_mean[:,1:], z_logstd[:,1:], prior_t1_mean, prior_t1_logstd).sum(dim=[1,-1])
        else:
            init_nll = -gaussian_log_prob(z_mean, z_logstd, z_sample)[:,1:].sum(dim=-1)
            z_sample, ldj = self.flow(z_sample.flatten(0, 1))
            z_sample = z_sample.unflatten(0, (imgs.shape[0], -1))
            ldj = ldj.unflatten(0, (imgs.shape[0], -1))[:,1:]
            # prior_t1_mean, prior_t1_logstd = self.prior_net(torch.cat([z_sample[:,:-1], target], dim=-1)).chunk(2, dim=-1)
            prior_t1_mean, prior_t1_logstd = self._run_prior(z_sample, target)
            out_nll = -gaussian_log_prob(prior_t1_mean, prior_t1_logstd, z_sample[:,1:]).sum(dim=-1)
            p_z = out_nll 
            p_z_x = init_nll - ldj
            kld = -(p_z_x - p_z).sum(dim=1)

        rec_loss = F.mse_loss(x_rec, labels[:,1:], reduction='none').sum(dim=list(range(2, len(x_rec.shape))))
        kld_factor = self.kld_scheduler.get_factor(self.global_step)
        loss = (kld_factor * kld * self.hparams.beta_t1 + rec_loss.sum(dim=1)).mean()
        loss = loss / (imgs.shape[1] - 1)

        self.log(f'{mode}_kld', kld.mean() / (imgs.shape[1]-1))
        self.log(f'{mode}_rec_loss', rec_loss.mean())
        if mode == 'train':
            self.log(f'{mode}_kld_scheduling', kld_factor)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='train')
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()
        loss = self.triplet_evaluation(batch, mode='val')
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        self.eval()
        loss = self.triplet_evaluation(batch, mode='test')
        self.log('test_loss', loss)

    def triplet_prediction(self, imgs, source):
        input_imgs = imgs[:,:2].flatten(0, 1)
        z_mean, z_logstd = self.encoder(input_imgs)
        if self.hparams.use_flow_prior:
            z_mean, _ = self.flow(z_mean)
        input_samples = z_mean
        input_samples = input_samples.unflatten(0, (-1, 2))
        
        target_assignment = self.target_assignment.to(z_mean.device)
        if target_assignment.shape[-1] > source.shape[-1]:
            target_assignment = target_assignment[...,:source.shape[-1]]
        mask_1 = (target_assignment[None,:,:] * (1 - source[:,None,:])).sum(dim=-1)
        mask_2 = 1 - mask_1
        triplet_samples = mask_1 * input_samples[:,0] + mask_2 * input_samples[:,1]
        if self.hparams.use_flow_prior:
            triplet_samples = self.flow.reverse(triplet_samples)
        triplet_rec = self.decoder(triplet_samples)
        if self.output_to_input is not None:
            triplet_rec = self.output_to_input(triplet_rec)
        return triplet_rec

    def triplet_evaluation(self, batch, mode='val'):
        if len(batch) == 2:
            imgs, source = batch
            labels = imgs
            latents = None
        elif len(batch[1].shape) == 2:
            imgs, source, latents = batch
            labels = imgs
        elif len(batch) == 3:
            imgs, labels, source = batch
            latents = None
        elif len(batch) == 4:
            imgs, labels, source, latents = batch
        triplet_label = labels[:,-1]
        triplet_rec = self.triplet_prediction(imgs, source)

        if self.causal_encoder is not None and latents is not None:
            self.causal_encoder.eval()
            with torch.no_grad():
                losses, dists, norm_dists, v_dict = self.causal_encoder.get_distances(triplet_rec, latents[:,-1], return_norm_dists=True, return_v_dict=True)
                rec_loss = sum([norm_dists[key].mean() for key in losses])
                mean_loss = sum([losses[key].mean() for key in losses])
                self.log(f'{mode}_distance_loss', mean_loss)
                for key in dists:
                    self.log(f'{mode}_{key}_dist', dists[key].mean())
                    self.log(f'{mode}_{key}_norm_dist', norm_dists[key].mean())
                    self.all_val_dists[key].append(dists[key])
                if self.current_epoch > 0 and self.causal_encoder_true_epoch >= self.current_epoch:
                    self.causal_encoder_true_epoch = self.current_epoch
                    if len(triplet_label.shape) == 2 and hasattr(self, 'autoencoder'):
                        triplet_label = self.autoencoder.decoder(triplet_label)
                    _, true_dists = self.causal_encoder.get_distances(triplet_label, latents[:,-1])
                    for key in dists:
                        self.log(f'{mode}_{key}_true_dist', true_dists[key].mean())
        else:
            rec_loss = 0.0

        return rec_loss

    def validation_epoch_end(self, *args, **kwargs):
        super().validation_epoch_end(*args, **kwargs)
        if len(self.all_val_dists.keys()) > 0:
            if self.current_epoch > 0 or True:
                means = {}
                for key in self.all_val_dists:
                    vals = torch.cat(self.all_val_dists[key], dim=0)
                    self.logger.experiment.add_histogram(key, vals, self.current_epoch)
                    means[key] = vals.mean().item()
                log_dict(d=means,
                         name='triplet_dists',
                         current_epoch=self.current_epoch,
                         log_dir=self.logger.log_dir)
            self.all_val_dists = defaultdict(list)

    @staticmethod
    def get_callbacks(exmp_inputs=None, dataset=None, cluster=False, correlation_dataset=None, correlation_test_dataset=None, **kwargs):
        img_callback = ImageLogCallback(exmp_inputs, dataset, every_n_epochs=10, cluster=cluster)
        corr_callback = BaselineCorrelationMetricsLogCallback(correlation_dataset, cluster=cluster, test_dataset=correlation_test_dataset)
        # Create learning rate callback
        lr_callback = LearningRateMonitor('step')
        return [lr_callback, img_callback, corr_callback]