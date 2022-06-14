"""
All models regarding CITRIS as PyTorch Lightning modules that we use for training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import OrderedDict, defaultdict
from tqdm.auto import tqdm

import sys
sys.path.append('../')
from models.shared import CosineWarmupScheduler, get_act_fn, ImageLogCallback, CorrelationMetricsLogCallback
from models.ae import Autoencoder
from models.citris_vae import CITRISVAE
from models.shared import AutoregNormalizingFlow


class CITRISNF(CITRISVAE):
    """ 
    The main module implementing CITRIS-NF.
    It is a subclass of CITRIS-VAE to inherit several functionality.
    """

    def __init__(self, *args,
                        autoencoder_checkpoint=None,
                        num_flows=4,
                        hidden_per_var=16,
                        num_samples=8,
                        flow_act_fn='silu',
                        noise_level=-1,
                        **kwargs):
        """
        Parameters
        ----------
        *args, **kwargs - see CITRIS-VAE for the full list
        autoencoder_checkpoint : str
                                 Path to the checkpoint of the autoencoder
                                 which should be used for training the flow
                                 on.
        num_flows : int
                    Number of flow layers to use
        hidden_per_var : int
                         Hidden dimensionality per latent variable to use
                         in the autoregressive networks.
        num_samples : int
                      Number of samples to take from an input encoding
                      during training. Larger sample sizes give smoother
                      gradients.
        flow_act_fn : str
                      Activation function to use in the networks of the flow
        noise_level : float
                      Standard deviation of the added noise to the encodings.
                      If smaller than zero, the std of the autoencoder is used.
        """
        kwargs['no_encoder_decoder'] = True  # We do not need any additional en- or decoder
        super().__init__(*args, **kwargs)
        # Initialize the flow
        self.flow = AutoregNormalizingFlow(self.hparams.num_latents, 
                                           self.hparams.num_flows,
                                           act_fn=get_act_fn(self.hparams.flow_act_fn),
                                           hidden_per_var=self.hparams.hidden_per_var)
        # Setup autoencoder
        self.autoencoder = Autoencoder.load_from_checkpoint(self.hparams.autoencoder_checkpoint)
        for p in self.autoencoder.parameters():
            p.requires_grad_(False)
        assert self.hparams.num_latents == self.autoencoder.hparams.num_latents, 'Autoencoder and flow need to have the same number of latents'
        assert self.autoencoder.hparams.img_width == self.causal_encoder.hparams.img_width, 'Autoencoder and Causal Encoder need to have the same image dimensions.'

        if self.hparams.noise_level < 0.0:
            self.hparams.noise_level = self.autoencoder.hparams.noise_level

    def encode(self, x, random=True):
        # Map input to disentangled latents, e.g. for correlation metrics
        if random:
            x = x + torch.randn_like(x) * self.hparams.noise_level
        z, ldj = self.flow(x)
        return z

    def _get_loss(self, batch, mode='train'):
        """ Main training method for calculating the loss """
        if len(batch) == 2:
            x_enc, target = batch
        else:
            x_enc, _, target = batch
        # Expand encodings over samples and add noise to 'sample' from the autoencoder
        # latent distribution
        x_enc = x_enc[...,None,:].expand(-1, -1, self.hparams.num_samples, -1)
        batch_size, seq_len, num_samples, num_latents = x_enc.shape
        x_sample = x_enc + torch.randn_like(x_enc) * self.hparams.noise_level
        x_sample = x_sample.flatten(0, 2)
        # Execute the flow
        z_sample, ldj = self.flow(x_sample)
        z_sample = z_sample.unflatten(0, (batch_size, seq_len, num_samples))
        ldj = ldj.reshape(batch_size, seq_len, num_samples)
        # Calculate the negative log likelihood of the transition prior
        nll = self.prior_t1.sample_based_nll(z_t=z_sample[:,:-1].flatten(0, 1),
                                             z_t1=z_sample[:,1:].flatten(0, 1),
                                             target=target.flatten(0, 1))
        # Add LDJ and prior NLL for full loss
        ldj = ldj[:,1:].flatten(0, 1).mean(dim=-1)  # Taking the mean over samples
        loss = nll + ldj
        loss = (loss * self.hparams.beta_t1 * (seq_len - 1)).mean()
        # Target classifier loss
        z_sample = z_sample.permute(0, 2, 1, 3).flatten(0, 1)  # Samples to batch dimension
        target = target[:,None].expand(-1, num_samples, -1, -1).flatten(0, 1)
        loss_model, loss_z = self.intv_classifier(z_sample=z_sample,
                                                  logger=self, 
                                                  target=target,
                                                  transition_prior=self.prior_t1)
        loss = loss + loss_model + loss_z * self.hparams.beta_classifier

        # Logging
        self.log(f'{mode}_nll', nll.mean())
        self.log(f'{mode}_ldj', ldj.mean())
        self.log(f'{mode}_intv_classifier_model', loss_model)
        self.log(f'{mode}_intv_classifier_z', loss_z)

        return loss

    def triplet_prediction(self, x_encs, source):
        """ Generates the triplet prediction of input encoding pairs and causal mask """
        batch_size = x_encs.shape[0]
        if isinstance(self.flow, nn.Identity):
            input_samples = self.flow(x_encs[:,:2].flatten(0, 1))
        else:
            input_samples, _ = self.flow(x_encs[:,:2].flatten(0, 1))
        input_samples = input_samples.unflatten(0, (-1, 2))
        # Map the causal mask to a latent variable mask
        target_assignment = self.prior_t1.get_target_assignment(hard=True)
        if source.shape[-1] + 1 == target_assignment.shape[-1]:  # No-variables missing
            source = torch.cat([source, source[...,-1:] * 0.0], dim=-1)
        elif target_assignment.shape[-1] > source.shape[-1]:
            target_assignment = target_assignment[...,:source.shape[-1]]
        # Take the latent variables from encoding 1 respective to the mask, and encoding 2 the inverse
        mask_1 = (target_assignment[None,:,:] * (1 - source[:,None,:])).sum(dim=-1)
        mask_2 = 1 - mask_1
        triplet_samples = mask_1 * input_samples[:,0] + mask_2 * input_samples[:,1]
        # Decode by reversing the flow, and using the pretrained decoder
        self.autoencoder.eval()  # Set to eval in any case
        triplet_samples = self.flow.reverse(triplet_samples)
        triplet_rec = self.autoencoder.decoder(triplet_samples)
        return triplet_rec

    @staticmethod
    def get_callbacks(exmp_inputs=None, dataset=None, cluster=False, correlation_dataset=False, correlation_test_dataset=None, **kwargs):
        img_callback = ImageLogCallback([None, None], dataset, every_n_epochs=10, cluster=cluster)
        corr_callback = CorrelationMetricsLogCallback(correlation_dataset, cluster=cluster, test_dataset=correlation_test_dataset)
        # Create learning rate callback
        lr_callback = LearningRateMonitor('step')
        return [lr_callback, img_callback, corr_callback]