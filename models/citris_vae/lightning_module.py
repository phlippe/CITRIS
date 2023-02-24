import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import numpy as np
import os
from collections import OrderedDict, defaultdict

import sys
sys.path.append('../')
from models.shared import CosineWarmupScheduler, SineWarmupScheduler, get_act_fn, log_dict, Encoder, Decoder, SimpleEncoder, SimpleDecoder, TransitionPrior, TargetClassifier, CausalEncoder, ImageLogCallback, CorrelationMetricsLogCallback
from models.shared import AutoregNormalizingFlow, gaussian_log_prob


class CITRISVAE(pl.LightningModule):
    """ The main module implementing CITRIS-VAE """

    def __init__(self, c_hid, num_latents, lr, 
                       num_causal_vars,
                       warmup=100, max_iters=100000,
                       kld_warmup=0,
                       imperfect_interventions=False,
                       img_width=64,
                       c_in=3,
                       lambda_reg=0.01,
                       var_names=None,
                       causal_encoder_checkpoint=None,
                       classifier_num_layers=1,
                       classifier_act_fn='silu',
                       classifier_gumbel_temperature=1.0,
                       classifier_use_normalization=True,
                       classifier_use_conditional_targets=True,
                       classifier_momentum=0.9,
                       decoder_num_blocks=1,
                       act_fn='silu',
                       no_encoder_decoder=False,
                       use_flow_prior=True,
                       cluster_logging=False,
                       **kwargs):
        """
        Parameters
        ----------
        c_hid : int
                Hidden dimensionality to use in the network
        num_latents : int
                      Number of latent variables in the VAE
        lr : float
             Learning rate to use for training
        num_causal_vars : int
                          Number of causal variables / size of intervention target vector
        warmup : int
                 Number of learning rate warmup steps
        max_iters : int
                    Number of max. training iterations. Needed for 
                    cosine annealing of the learning rate.
        kld_warmup : int
                     Number of steps in the KLD warmup (default no warmup)
        imperfect_interventions : bool
                                  Whether interventions can be imperfect or not
        img_width : int
                    Width of the input image (assumed to be equal to height)
        c_in : int
               Number of input channels (3 for RGB)
        lambda_reg : float
                     Regularizer for promoting intervention-independent information to be modeled
                     in psi(0)
        var_names : list
                    List of names of the causal variables. Used for logging.
        causal_encoder_checkpoint : str
                                    Path to the checkpoint of a Causal-Encoder model to use for
                                    the triplet evaluation.
        classifier_num_layers : int
                                Number of layers to use in the target classifier network.
        classifier_act_fn : str
                            Activation function to use in the target classifier network.
        classifier_gumbel_temperature : float
                                        Temperature to use for the Gumbel Softmax sampling.
        classifier_use_normalization : bool
                                       Whether to use LayerNorm in the target classifier or not.
        classifier_use_conditional_targets : bool
                                             If True, we record conditional targets p(I^t+1_i|I^t+1_j)
                                             in the target classifier. Needed when intervention targets 
                                             are confounded.
        classifier_momentum_model : float
                                    Whether to use momentum or not in smoothing the target classifier
        decoder_num_blocks : int
                             Number of residual blocks to use per dimension in the decoder.
        act_fn : str
                 Activation function to use in the encoder and decoder network.
        no_encoder_decoder : bool
                             If True, no encoder or decoder are initialized. Used for CITRIS-NF
        """
        super().__init__()
        self.save_hyperparameters()
        act_fn_func = get_act_fn(self.hparams.act_fn)

        # Encoder-Decoder init
        if self.hparams.no_encoder_decoder:
            self.encoder, self.decoder = nn.Identity(), nn.Identity()
        else:
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
                                          variational=True)
                self.decoder = Decoder(num_latents=self.hparams.num_latents,
                                          c_hid=self.hparams.c_hid,
                                          c_out=self.hparams.c_in,
                                          width=self.hparams.img_width,
                                          num_blocks=self.hparams.decoder_num_blocks,
                                          act_fn=act_fn_func)
        # Transition prior
        self.prior_t1 = TransitionPrior(num_latents=self.hparams.num_latents,
                                        num_blocks=self.hparams.num_causal_vars,
                                        c_hid=self.hparams.c_hid,
                                        imperfect_interventions=self.hparams.imperfect_interventions,
                                        lambda_reg=self.hparams.lambda_reg,
                                        autoregressive_model=self.hparams.autoregressive_prior,
                                        gumbel_temperature=self.hparams.classifier_gumbel_temperature)
        # Target classifier
        self.intv_classifier = TargetClassifier(num_latents=self.hparams.num_latents,
                                                num_blocks=self.hparams.num_causal_vars,
                                                c_hid=self.hparams.c_hid,
                                                num_layers=self.hparams.classifier_num_layers,
                                                act_fn=get_act_fn(self.hparams.classifier_act_fn),
                                                var_names=self.hparams.var_names,
                                                momentum_model=self.hparams.classifier_momentum,
                                                gumbel_temperature=self.hparams.classifier_gumbel_temperature,
                                                use_normalization=self.hparams.classifier_use_normalization,
                                                use_conditional_targets=self.hparams.classifier_use_conditional_targets)

        if self.hparams.use_flow_prior:
            self.flow = AutoregNormalizingFlow(self.hparams.num_latents,
                                               num_flows=4,
                                               act_fn=nn.SiLU,
                                               hidden_per_var=16)

        # Warmup scheduler for KL (if selected)
        self.kld_scheduler = SineWarmupScheduler(kld_warmup)
        # Load causal encoder for triplet evaluation
        if self.hparams.causal_encoder_checkpoint is not None:
            self.causal_encoder_true_epoch = int(1e5)  # We want to log the true causal encoder distance once
            self.causal_encoder = CausalEncoder.load_from_checkpoint(self.hparams.causal_encoder_checkpoint)
            for p in self.causal_encoder.parameters():
                p.requires_grad_(False)
        else:
            self.causal_encoder = None
        # Logging
        self.all_val_dists = defaultdict(list)
        self.all_v_dicts = []
        self.output_to_input = None

    def forward(self, x):
        # Full encoding and decoding of samples
        z_mean, z_logstd = self.encoder(x)
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        x_rec = self.decoder(z_sample)
        return x_rec, z_sample, z_mean, z_logstd

    def encode(self, x, random=True):
        # Map input to encoding, e.g. for correlation metrics
        z_mean, z_logstd = self.encoder(x)
        if random:
            z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        else:
            z_sample = z_mean
        if self.hparams.use_flow_prior:
            z_sample, _ = self.flow(z_sample)
        return z_sample

    def configure_optimizers(self):
        # We use different learning rates for the target classifier (higher lr for faster learning).
        intv_params, other_params = [], []
        for name, param in self.named_parameters():
            if name.startswith('intv_classifier'):
                intv_params.append(param)
            else:
                other_params.append(param)
        optimizer = optim.AdamW([{'params': intv_params, 'lr': self.hparams.classifier_lr, 'weight_decay': 1e-4},
                                 {'params': other_params}], lr=self.hparams.lr, weight_decay=0.0)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=self.hparams.warmup,
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _get_loss(self, batch, mode='train'):
        """ Main training method for calculating the loss """
        if len(batch) == 2:
            imgs, target = batch
            labels = imgs
        else:
            imgs, labels, target = batch
        # En- and decode every element of the sequence, except first element no decoding
        z_mean, z_logstd = self.encoder(imgs.flatten(0, 1))
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        x_rec = self.decoder(z_sample.unflatten(0, imgs.shape[:2])[:,1:].flatten(0, 1))
        z_sample, z_mean, z_logstd, x_rec = [t.unflatten(0, (imgs.shape[0], -1)) for t in [z_sample, z_mean, z_logstd, x_rec]]

        if self.hparams.use_flow_prior:
            init_nll = -gaussian_log_prob(z_mean[:,1:], z_logstd[:,1:], z_sample[:,1:]).sum(dim=-1)
            z_sample, ldj = self.flow(z_sample.flatten(0, 1))
            z_sample = z_sample.unflatten(0, (imgs.shape[0], -1))
            ldj = ldj.unflatten(0, (imgs.shape[0], -1))[:,1:]
            out_nll = self.prior_t1.sample_based_nll(z_t1=z_sample[:,None,1:].flatten(0, 1), 
                                                     target=target.flatten(0, 1), 
                                                      z_t=z_sample[:,None,:-1].flatten(0, 1))
            out_nll = out_nll.unflatten(0, (imgs.shape[0], -1))
            p_z = out_nll 
            p_z_x = init_nll - ldj
            kld = -(p_z_x - p_z)
            kld_t1_all = kld.unflatten(0, (imgs.shape[0], -1)).sum(dim=1)
        else:
            # Calculate KL divergence between every pair of frames
            kld_t1_all = self.prior_t1.kl_divergence(z_t=z_mean[:,:-1].flatten(0, 1), 
                                                     target=target.flatten(0, 1), 
                                                     z_t1_mean=z_mean[:,1:].flatten(0, 1), 
                                                     z_t1_logstd=z_logstd[:,1:].flatten(0, 1), 
                                                     z_t1_sample=z_sample[:,1:].flatten(0, 1))
            kld_t1_all = kld_t1_all.unflatten(0, (imgs.shape[0], -1)).sum(dim=1)

        # Calculate reconstruction loss
        if isinstance(self.decoder, nn.Identity):
            rec_loss = z_mean.new_zeros(imgs.shape[0], imgs.shape[1])
        else:
            rec_loss = F.mse_loss(x_rec, labels[:,1:], reduction='none').sum(dim=[-3, -2, -1])
        # Combine to full loss
        kld_factor = self.kld_scheduler.get_factor(self.global_step)
        loss = (kld_factor * (kld_t1_all * self.hparams.beta_t1) + rec_loss.sum(dim=1)).mean()
        loss = loss / (imgs.shape[1] - 1)
        # Add target classifier loss
        loss_model, loss_z = self.intv_classifier(z_sample=z_sample,
                                                  logger=self if not self.hparams.cluster_logging else None, 
                                                  target=target,
                                                  transition_prior=self.prior_t1)
        loss = loss + (loss_model + loss_z) * self.hparams.beta_classifier

        # Logging
        self.log(f'{mode}_kld_t1', kld_t1_all.mean() / (imgs.shape[1]-1))
        self.log(f'{mode}_rec_loss_t1', rec_loss.mean())
        if mode == 'train':
            self.log(f'{mode}_kld_scheduling', kld_factor)
        self.log(f'{mode}_intv_classifier_model', loss_model)
        self.log(f'{mode}_intv_classifier_z', loss_z)

        return loss

    def triplet_prediction(self, imgs, source):
        """ Generates the triplet prediction of input image pairs and causal mask """
        input_imgs = imgs[:,:2].flatten(0, 1)
        z_mean, z_logstd = self.encoder(input_imgs)
        if self.hparams.use_flow_prior:
            z_mean, _ = self.flow(z_mean)
        input_samples = z_mean
        input_samples = input_samples.unflatten(0, (-1, 2))
        # Map the causal mask to a latent variable mask
        target_assignment = self.prior_t1.get_target_assignment(hard=True)
        if source.shape[-1] + 1 == target_assignment.shape[-1]:  # No-variables missing
            source = torch.cat([source, source[...,-1:] * 0.0], dim=-1)
        elif target_assignment.shape[-1] > source.shape[-1]:
            target_assignment = target_assignment[...,:source.shape[-1]]
        # Take the latent variables from image 1 respective to the mask, and image 2 the inverse
        mask_1 = (target_assignment[None,:,:] * (1 - source[:,None,:])).sum(dim=-1)
        mask_2 = 1 - mask_1
        triplet_samples = mask_1 * input_samples[:,0] + mask_2 * input_samples[:,1]
        if self.hparams.use_flow_prior:
            triplet_samples = self.flow.reverse(triplet_samples)
        # Decode the new combination
        triplet_rec = self.decoder(triplet_samples)
        if self.output_to_input is not None:
            triplet_rec = self.output_to_input(triplet_rec)
        return triplet_rec

    def triplet_evaluation(self, batch, mode='val'):
        """ Evaluates the triplet prediction for a batch of images """
        # Input handling
        if len(batch) == 2:
            imgs, source = batch
            labels = imgs
            latents = None
            obj_indices = None
        elif len(batch) == 3 and len(batch[1].shape) == 2:
            imgs, source, latents = batch
            labels = imgs
            obj_indices = None
        elif len(batch) == 3:
            imgs, labels, source = batch
            obj_indices = None
            latents = None
        elif len(batch) == 4 and len(batch[-1].shape) > 1:
            imgs, labels, source, latents = batch
            obj_indices = None
        elif len(batch) == 4 and len(batch[-1].shape) == 1:
            imgs, source, latents, obj_indices = batch
            labels = imgs
        triplet_label = labels[:,-1]
        # Estimate triplet prediction
        triplet_rec = self.triplet_prediction(imgs, source)

        if self.causal_encoder is not None and latents is not None:
            self.causal_encoder.eval()
            # Evaluate the causal variables of the predicted output
            with torch.no_grad():
                losses, dists, norm_dists, v_dict = self.causal_encoder.get_distances(triplet_rec, latents[:,-1], return_norm_dists=True, return_v_dict=True)
                self.all_v_dicts.append(v_dict)
                rec_loss = sum([norm_dists[key].mean() for key in losses])
                mean_loss = sum([losses[key].mean() for key in losses])
                self.log(f'{mode}_distance_loss', mean_loss)
                for key in dists:
                    self.all_val_dists[key].append(dists[key])
                    self.log(f'{mode}_{key}_dist', dists[key].mean())
                    self.log(f'{mode}_{key}_norm_dist', norm_dists[key].mean())
                    if obj_indices is not None:  # For excluded object shapes, record results separately
                        for v in obj_indices.unique().detach().cpu().numpy():
                            self.log(f'{mode}_{key}_dist_obj_{v}', dists[key][obj_indices == v].mean())
                            self.log(f'{mode}_{key}_norm_dist_obj_{v}', norm_dists[key][obj_indices == v].mean())
                if obj_indices is not None:
                    self.all_val_dists['object_indices'].append(obj_indices)
                if self.current_epoch > 0 and self.causal_encoder_true_epoch >= self.current_epoch:
                    self.causal_encoder_true_epoch = self.current_epoch
                    if len(triplet_label.shape) == 2 and hasattr(self, 'autoencoder'):
                        triplet_label = self.autoencoder.decoder(triplet_label)
                    _, true_dists = self.causal_encoder.get_distances(triplet_label, latents[:,-1])
                    for key in dists:
                        self.log(f'{mode}_{key}_true_dist', true_dists[key].mean())
        else:
            rec_loss = torch.zeros(1,)

        return rec_loss

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='train')
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, *_ = batch
        loss = self.triplet_evaluation(batch, mode='val')
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        imgs, *_ = batch
        loss = self.triplet_evaluation(batch, mode='test')
        self.log('test_loss', loss)

    def validation_epoch_end(self, *args, **kwargs):
        # Logging at the end of validation
        super().validation_epoch_end(*args, **kwargs)
        if len(self.all_val_dists.keys()) > 0:
            if self.current_epoch > 0:
                means = {}
                if 'object_indices' in self.all_val_dists:
                    obj_indices = torch.cat(self.all_val_dists['object_indices'], dim=0)
                    unique_objs = obj_indices.unique().detach().cpu().numpy().tolist()
                for key in self.all_val_dists:
                    if key == 'object_indices':
                        continue
                    vals = torch.cat(self.all_val_dists[key], dim=0)
                    if 'object_indices' in self.all_val_dists:
                        for o in unique_objs:
                            sub_vals = vals[obj_indices == o]
                            key_obj = key + f'_obj_{o}'
                            self.logger.experiment.add_histogram(key_obj, sub_vals, self.current_epoch)
                            means[key_obj] = sub_vals.mean().item()
                    else:
                        self.logger.experiment.add_histogram(key, vals, self.current_epoch)
                        means[key] = vals.mean().item()
                log_dict(d=means,
                         name='triplet_dists',
                         current_epoch=self.current_epoch,
                         log_dir=self.logger.log_dir)
            self.all_val_dists = defaultdict(list)
        if len(self.all_v_dicts) > 0:
            outputs = {}
            for key in self.all_v_dicts[0]:
                outputs[key] = torch.cat([v[key] for v in self.all_v_dicts], dim=0).cpu().numpy()
            np.savez_compressed(os.path.join(self.logger.log_dir, 'causal_encoder_v_dicts.npz'), **outputs)
            self.all_v_dicts = []

    @staticmethod
    def get_callbacks(exmp_inputs=None, dataset=None, cluster=False, correlation_dataset=None, correlation_test_dataset=None, **kwargs):
        img_callback = ImageLogCallback(exmp_inputs, dataset, every_n_epochs=10 if not cluster else 50, cluster=cluster)
        corr_callback = CorrelationMetricsLogCallback(correlation_dataset, cluster=cluster, test_dataset=correlation_test_dataset)
        # Create learning rate callback
        lr_callback = LearningRateMonitor('step')
        return [lr_callback, img_callback, corr_callback]