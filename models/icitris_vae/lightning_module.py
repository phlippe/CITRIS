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
from models.shared import CosineWarmupScheduler, SineWarmupScheduler, gaussian_log_prob, get_act_fn, log_dict, Encoder, Decoder, SimpleEncoder, SimpleDecoder, CausalEncoder, CorrelationMetricsLogCallback, ImageLogCallback, TargetClassifier, GraphLogCallback, SparsifyingGraphCallback
from models.icitris_vae.prior import InstantaneousPrior
from models.icitris_vae.target_classifier import InstantaneousTargetClassifier
from models.icitris_vae.mi_estimator import MIEstimator
from models.shared import AutoregNormalizingFlow


class iCITRISVAE(pl.LightningModule):
    """ The main module implementing iCITRIS-VAE """

    def __init__(self, c_hid, num_latents, lr, 
                       num_causal_vars,
                       warmup=100, max_iters=100000,
                       img_width=64,
                       graph_learning_method="ENCO",
                       graph_lr=5e-4,
                       c_in=3,
                       lambda_sparse=0.0,
                       lambda_reg=0.01,
                       num_graph_samples=8,
                       causal_encoder_checkpoint=None,
                       act_fn='silu',
                       beta_classifier=2.0,
                       beta_mi_estimator=2.0,
                       no_encoder_decoder=False,
                       var_names=None,
                       autoregressive_prior=False,
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
        img_width : int
                    Width of the input image (assumed to be equal to height)
        graph_learning_method : str
                                Which graph learning method to use in the prior.
                                Options: ENCO, NOTEARS
        graph_lr : float
                   Learning rate of the graph parameters
        c_in : int
               Number of input channels (3 for RGB)
        lambda_sprase : float
                        Regularizer for encouraging sparse graphs
        lambda_reg : float
                     Regularizer for promoting intervention-independent information to be modeled
                     in psi(0)
        num_graph_samples : int
                            Number of graph samples to use in ENCO's gradient estimation
        beta_classifier : float
                          Weight of the target classifier in training
        beta_mi_estimator : float
                            Weight of the mutual information estimator in training
        causal_encoder_checkpoint : str
                                    Path to the checkpoint of a Causal-Encoder model to use for
                                    the triplet evaluation.
        act_fn : str
                 Activation function to use in the encoder and decoder network.
        no_encoder_decoder : bool
                             If True, no encoder or decoder are initialized. Used for CITRIS-NF
        var_names : Optional[List[str]]
                    Names of the causal variables, for plotting and logging
        autoregressive_prior : bool
                               If True, the prior per causal variable is autoregressive
        use_flow_prior : bool
                         If True, use a NF prior in the VAE.
        cluster_logging : bool
                          If True, the logging will be reduced to a minimum
        """
        super().__init__()
        self.save_hyperparameters()
        act_fn_func = get_act_fn(self.hparams.act_fn)

        # Encoder-Decoder init
        if not self.hparams.no_encoder_decoder:
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
                                          num_blocks=1,
                                          act_fn=act_fn_func)
        else:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()
        # Prior
        self.prior = InstantaneousPrior(num_latents=self.hparams.num_latents,
                                        c_hid=self.hparams.c_hid,
                                        num_blocks=self.hparams.num_causal_vars,
                                        shared_inputs=self.hparams.num_latents,
                                        num_graph_samples=self.hparams.num_graph_samples,
                                        lambda_sparse=self.hparams.lambda_sparse,
                                        graph_learning_method=self.hparams.graph_learning_method,
                                        autoregressive=self.hparams.autoregressive_prior)
        self.intv_classifier = InstantaneousTargetClassifier(
                                                num_latents=self.hparams.num_latents,
                                                num_blocks=self.hparams.num_causal_vars,
                                                c_hid=self.hparams.c_hid*2,
                                                num_layers=1,
                                                act_fn=nn.SiLU,
                                                var_names=self.hparams.var_names,
                                                momentum_model=0.9,
                                                gumbel_temperature=1.0,
                                                use_normalization=True,
                                                use_conditional_targets=True)
        self.mi_estimator = MIEstimator(num_latents=self.hparams.num_latents,
                                        num_blocks=self.hparams.num_causal_vars,
                                        c_hid=self.hparams.c_hid,
                                        var_names=self.hparams.var_names,
                                        momentum_model=0.9,
                                        gumbel_temperature=1.0)
        if self.hparams.use_flow_prior:
            self.flow = AutoregNormalizingFlow(self.hparams.num_latents,
                                               num_flows=4,
                                               act_fn=nn.SiLU,
                                               hidden_per_var=16)

        self.mi_scheduler = SineWarmupScheduler(warmup=50000,
                                                start_factor=0.004,
                                                end_factor=1.0,
                                                offset=20000)
        self.matrix_exp_scheduler = SineWarmupScheduler(warmup=100000,
                                                        start_factor=-6,
                                                        end_factor=4,
                                                        offset=10000)

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
        self.prior_t1 = self.prior

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
        graph_params, counter_params, other_params = [], [], []
        for name, param in self.named_parameters():
            if name.startswith('prior.enco') or name.startswith('prior.notears'):
                graph_params.append(param)
            elif name.startswith('intv_classifier') or name.startswith('mi_estimator'):
                counter_params.append(param)
            else:
                other_params.append(param)
        optimizer = optim.AdamW([{'params': graph_params, 'lr': self.hparams.graph_lr, 'weight_decay': 0.0, 'eps': 1e-8},
                                 {'params': counter_params, 'lr': 2*self.hparams.lr, 'weight_decay': 1e-4},
                                 {'params': other_params}], lr=self.hparams.lr, weight_decay=0.0)
        lr_scheduler = CosineWarmupScheduler(optimizer,
                                             warmup=[200*self.hparams.warmup, 2*self.hparams.warmup, 2*self.hparams.warmup],
                                             offset=[10000, 0, 0],
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _get_loss(self, batch, mode='train'):
        """ Main training method for calculating the loss """
        if len(batch) == 2:
            imgs, target = batch
            labels = imgs
        else:
            imgs, labels, target = batch
        # En- and decode every element
        z_mean, z_logstd = self.encoder(imgs.flatten(0, 1))
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        z_sample = z_sample.unflatten(0, imgs.shape[:2])
        z_sample[:,0] = z_mean.unflatten(0, imgs.shape[:2])[:,0]
        z_sample = z_sample.flatten(0, 1)
        x_rec = self.decoder(z_sample.unflatten(0, imgs.shape[:2])[:,1:].flatten(0, 1))
        z_sample, z_mean, z_logstd, x_rec = [t.unflatten(0, (imgs.shape[0], -1)) for t in [z_sample, z_mean, z_logstd, x_rec]]

        if self.hparams.use_flow_prior:
            init_nll = -gaussian_log_prob(z_mean[:,1:], z_logstd[:,1:], z_sample[:,1:]).sum(dim=-1)
            z_sample, ldj = self.flow(z_sample.flatten(0, 1))
            z_sample = z_sample.unflatten(0, (imgs.shape[0], -1))
            ldj = ldj.unflatten(0, (imgs.shape[0], -1))[:,1:]
            out_nll = self.prior.forward(z_sample=z_sample[:,1:].flatten(0, 1), 
                                         target=target.flatten(0, 1), 
                                         z_shared=z_sample[:,:-1].flatten(0, 1),
                                         matrix_exp_factor=np.exp(self.matrix_exp_scheduler.get_factor(self.global_step)))
            out_nll = out_nll.unflatten(0, (imgs.shape[0], -1))
            p_z = out_nll 
            p_z_x = init_nll - ldj
            kld = -(p_z_x - p_z)
            kld = kld.unflatten(0, (imgs.shape[0], -1))
        else:
            # Calculate KL divergence between every pair of frames
            kld = self.prior.forward(z_sample=z_sample[:,1:].flatten(0, 1), 
                                     z_mean=z_mean[:,1:].flatten(0, 1), 
                                     z_logstd=z_logstd[:,1:].flatten(0, 1), 
                                     target=target.flatten(0, 1), 
                                     z_shared=z_sample[:,:-1].flatten(0, 1),
                                     matrix_exp_factor=np.exp(self.matrix_exp_scheduler.get_factor(self.global_step)))
            kld = kld.unflatten(0, (imgs.shape[0], -1))
        
        # Calculate reconstruction loss
        rec_loss = F.mse_loss(x_rec, labels[:,1:], reduction='none').sum(dim=list(range(2, len(x_rec.shape))))
        # Combine to full loss
        loss = (kld * self.hparams.beta_t1 + rec_loss).mean()
        
        # Target classifier
        loss_model, loss_z = self.intv_classifier(z_sample=z_sample,
                                                  logger=self if not self.hparams.cluster_logging else None, 
                                                  target=target,
                                                  transition_prior=self.prior)
        loss = loss + loss_model + loss_z * self.hparams.beta_classifier

        # Mutual information estimator
        scheduler_factor = self.mi_scheduler.get_factor(self.global_step)
        loss_model_mi, loss_z_mi = self.mi_estimator(z_sample=z_sample,
                                                     logger=self if not self.hparams.cluster_logging else None, 
                                                     target=target,
                                                     transition_prior=self.prior,
                                                     instant_prob=scheduler_factor)
        loss = loss + loss_model_mi + loss_z_mi * self.hparams.beta_mi_estimator * (1.0 + 2.0 * scheduler_factor)
        
        # For stabilizing the mean, since it is unconstrained
        loss_z_reg = (z_sample.mean(dim=[0,1]) ** 2 + z_sample.std(dim=[0,1]).log() ** 2).mean()
        loss = loss + 0.1 * loss_z_reg 

        # Logging
        self.log(f'{mode}_kld', kld.mean())
        self.log(f'{mode}_rec_loss_t1', rec_loss.mean())
        if not self.hparams.cluster_logging:
            self.prior.logging(self)
        self.log(f'{mode}_intv_classifier_model', loss_model)
        self.log(f'{mode}_intv_classifier_z', loss_z)
        self.log(f'{mode}_mi_estimator_model', loss_model_mi)
        self.log(f'{mode}_mi_estimator_z', loss_z_mi)
        self.log(f'{mode}_mi_estimator_factor', scheduler_factor)
        self.log(f'{mode}_reg_loss', loss_z_reg)

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
        target_assignment = self.prior.get_target_assignment(hard=True)
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

    def training_epoch_end(self, *args, **kwargs):
        super().training_epoch_end(*args, **kwargs)
        self.prior.check_trainability()

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
        img_callback = ImageLogCallback(exmp_inputs, dataset, every_n_epochs=5 if not cluster else 50, cluster=cluster)
        corr_callback = CorrelationMetricsLogCallback(correlation_dataset, cluster=cluster, test_dataset=correlation_test_dataset)
        graph_callback = GraphLogCallback(every_n_epochs=(1 if not cluster else 5), dataset=dataset, cluster=cluster)
        sparse_graph_callback = SparsifyingGraphCallback(dataset=dataset, cluster=cluster)
        lr_callback = LearningRateMonitor('step')
        return [lr_callback, img_callback, corr_callback, graph_callback, sparse_graph_callback]
