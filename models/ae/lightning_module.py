import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('../../')
from models.shared import CosineWarmupScheduler, Encoder, Decoder, visualize_ae_reconstruction


class Autoencoder(pl.LightningModule):
    """ Simple Autoencoder network for (i)CITRIS-NF """

    def __init__(self, num_latents,
                       c_in=3,
                       c_hid=64,
                       lr=1e-3,
                       warmup=500, 
                       max_iters=100000,
                       img_width=64,
                       noise_level=0.05,
                       regularizer_weight=1e-4,
                       **kwargs):
        """
        Parameters
        ----------
        num_latents : int
                      Number of latents in the bottleneck.
        c_in : int
               Number of input channels (3 for RGB)
        c_hid : int
                Hidden dimensionality to use in the network
        lr : float
             Learning rate to use for training.
        warmup : int
                 Number of learning rate warmup steps
        max_iters : int
                    Number of max. training iterations. Needed for 
                    cosine annealing of the learning rate.
        img_width : int
                    Width of the input image (assumed to be equal to height)
        noise_level : float
                      Standard deviation of the added noise to the latents.
        """
        super().__init__()
        self.save_hyperparameters()

        self.encoder = Encoder(num_latents=self.hparams.num_latents,
                               c_hid=self.hparams.c_hid,
                               c_in=self.hparams.c_in,
                               width=self.hparams.img_width,
                               act_fn=nn.SiLU,
                               variational=False)
        self.decoder = Decoder(num_latents=self.hparams.num_latents,
                               c_hid=self.hparams.c_hid,
                               c_out=self.hparams.c_in,
                               width=self.hparams.img_width,
                               num_blocks=2,
                               act_fn=nn.SiLU)

    def forward(self, x, return_z=False):
        z = self.encoder(x)
        # Adding noise to latent encodings preventing potential latent space collapse
        z_samp = z + torch.randn_like(z) * self.hparams.noise_level
        x_rec = self.decoder(z_samp)
        if return_z:
            return x_rec, z
        else:
            return x_rec 

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.0)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _get_loss(self, batch, mode='train'):
        # Trained by standard MSE loss
        imgs = batch
        x_rec, z = self.forward(imgs, return_z=True)
        loss_rec = F.mse_loss(x_rec, imgs)
        loss_reg = (z ** 2).mean()
        self.log(f'{mode}_loss_rec', loss_rec)
        self.log(f'{mode}_loss_reg', loss_reg)
        self.log(f'{mode}_loss_reg_weighted', loss_reg * self.hparams.regularizer_weight)
        loss = loss_rec + loss_reg * self.hparams.regularizer_weight
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='train')
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='val')
        self.log('val_loss', loss)

    @staticmethod
    def get_callbacks(exmp_inputs=None, cluster=False, **kwargs):
        img_callback = AELogCallback(exmp_inputs, every_n_epochs=50)
        # Create learning rate callback
        lr_callback = LearningRateMonitor('step')
        return [lr_callback, img_callback]


class AELogCallback(pl.Callback):
    """ Callback for visualizing predictions """

    def __init__(self, exmp_inputs, every_n_epochs=5, prefix=''):
        super().__init__()
        self.imgs = exmp_inputs
        self.every_n_epochs = every_n_epochs
        self.prefix = prefix

    def on_train_epoch_end(self, trainer, pl_module):
        def log_fig(tag, fig):
            trainer.logger.experiment.add_figure(f'{self.prefix}{tag}', fig, global_step=trainer.global_step)
            plt.close(fig)

        if self.imgs is not None and (trainer.current_epoch+1) % self.every_n_epochs == 0:
            images = self.imgs.to(trainer.model.device)
            trainer.model.eval()
            log_fig(f'reconstruction_seq', visualize_ae_reconstruction(trainer.model, images[:8]))
            rand_idxs = np.random.permutation(images.shape[0])
            log_fig(f'reconstruction_rand', visualize_ae_reconstruction(trainer.model, images[rand_idxs[:8]]))
            trainer.model.train()