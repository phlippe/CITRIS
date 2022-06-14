import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

import sys
sys.path.append('../../')
from models.shared import kl_divergence, gaussian_log_prob, MultivarLinear, MultivarLayerNorm, MultivarStableTanh, AutoregLinear



class InstantaneousPrior(nn.Module):
    """ The prior of iCITRIS """

    def __init__(self, num_latents, 
                       c_hid, 
                       num_blocks, 
                       shared_inputs=0,
                       lambda_sparse=0.02,
                       gumbel_temperature=1.0,
                       num_graph_samples=8,
                       graph_learning_method='ENCO',
                       autoregressive=False):
        """
        For simplicity, this prior considers that zpsi{0} is an empty set.
        However, it can be easily extended by adding an extra causal dimension
        with set orientations.

        Parameters
        ----------
        num_latents : int
                      Number of latent variables
        c_hid : int
                Hidden dimension in the networks
        shared_inputs : int
                        Number of dimensions to condition the prior on.
                        Usually number of latents, i.e. the previous time step.
        lambda_sparse : float
                        Sparsity regularizer in the causal discovery method
        gumbel_temperature : float
                             Temperature for the gumbel softmax operations
        num_graph_samples : int
                            Number of graph samples to use in ENCO's gradient estimation
        graph_learning_method : str
                                Method to use for graph learning. Options: ENCO, NOTEARS
        autoregressive : bool
                         If True, the prior within a causal variable is autoregressive or not
        """
        super().__init__()
        self.num_latents = num_latents
        self.num_blocks = num_blocks
        self.shared_inputs = shared_inputs
        self.gumbel_temperature = gumbel_temperature
        self.graph_learning_method = graph_learning_method
        self.lambda_sparse = lambda_sparse
        self.autoregressive = autoregressive
        self.num_graph_samples = num_graph_samples
        
        self.target_params = nn.Parameter(torch.zeros(num_latents, num_blocks))
        self.net = nn.Sequential(
                MultivarLinear(self.num_latents * 2 + self.num_blocks + self.shared_inputs, c_hid, [self.num_latents]),
                MultivarLayerNorm(c_hid, [self.num_latents]),
                nn.SiLU(),
                MultivarLinear(c_hid, c_hid, [self.num_latents]),
                MultivarLayerNorm(c_hid, [self.num_latents]),
                nn.SiLU(),
                MultivarLinear(c_hid, 2, 
                               [self.num_latents]),
                MultivarStableTanh(2, [self.num_latents])
            )
        self.net[-2].weight.data.fill_(0.0)

        if self.graph_learning_method == 'ENCO':
            self.enco_theta = nn.Parameter(torch.zeros(num_blocks, num_blocks))
            self.enco_theta.data.masked_fill_(torch.eye(self.enco_theta.shape[0], dtype=torch.bool), -9e15)
            self.enco_gamma = nn.Parameter(self.enco_theta.data.clone())
            self.enco_gamma.data.masked_fill_(~torch.eye(self.enco_theta.shape[0], dtype=torch.bool), 4)
        elif self.graph_learning_method == 'NOTEARS':
            self.notears_params = nn.Parameter(torch.zeros(num_blocks, num_blocks))
            self.notears_params.data.masked_fill_(torch.eye(self.notears_params.shape[0], dtype=torch.bool), -9e15)
            self.num_graph_samples = 1
        else:
            assert False, f'Unknown graph learning method \"{self.graph_learning_method}\"'

        self.train_graph = True
        self.gradient_efficient = True

    def forward(self, z_sample, target, z_mean=None, z_logstd=None, num_graph_samples=-1, z_shared=None, matrix_exp_factor=0.0):
        """
        Calculate the NLL or KLD of samples under an instantaneous prior.

        Parameters
        ----------
        z_sample : torch.FloatTensor, shape [Batch_size, num_latents] or [Batch_size, num_samples, num_latents]
                   The latents of the time step for which the NLL/KLD should be calculated, i.e. z^t+1
        target : torch.FloatTensor, shape [Batch_size, num_causal_vars] or torch.LongTensor, shape [Batch_size]
                 The intervention targets to consider on C^t+1
        z_mean : Optional, torch.FloatTensor, shape [Batch_size, num_latents]
                 If given, the prior calculates the KL divergence to the distribution of a Gaussian
                 with mean z_mean and standard deviation exp(z_logstd)
        z_logstd : Optional, torch.FloatTensor, shape [Batch_size, num_latents]
                   See z_mean
        num_graph_samples : int
                            Number of graph samples to consider for estimating the gradients of 
                            ENCO. If smaller than zero, the default number of samples in 
                            self.num_graph_samples is used.
        z_shared : Optional, torch.FloatTensor, shape [Batch_size, shared_inputs]
                   The additional latents to consider in the prior (usually z^t)
        matrix_exp_factor : float
                            The weight of the acyclicity regularizer of NOTEARS
        """
        batch_size = z_sample.shape[0]
        if len(z_sample.shape) == 2:
            z_sample = z_sample[:, None]  # Add a 'sample' dimension
            if z_shared is not None:
                z_shared = z_shared[:, None]
        num_z_samples = z_sample.shape[1]
        num_graph_samples = self.num_graph_samples if num_graph_samples <= 0 else num_graph_samples

        # Target preparation
        if len(target.shape) == 1:
            target_oh = F.one_hot(target, num_classes=self.num_blocks)
        else:
            target_oh = target
        target_probs = torch.softmax(self.target_params, dim=-1)
        target_samples = F.gumbel_softmax(self.target_params[None].expand(target.shape[0], -1, -1), 
                                          tau=self.gumbel_temperature, hard=True)

        # Graph sampling
        if self.train_graph:  # When the graph parameters converged, we can stop the graph training for efficiency
            if self.graph_learning_method == 'ENCO':
                edge_prob = torch.sigmoid(self.enco_theta.detach()) * torch.sigmoid(self.enco_gamma.detach())
                adj_matrix = torch.bernoulli(edge_prob[None, None].expand(batch_size, num_graph_samples, -1, -1))
            else:
                edge_logits = torch.stack([torch.zeros_like(self.notears_params), self.notears_params], dim=-1)
                adj_matrix = F.gumbel_softmax(edge_logits[None, None].expand(batch_size, num_graph_samples, -1, -1, -1),
                                              tau=self.gumbel_temperature, hard=True)[...,1]
        else:
            adj_matrix = self.get_adj_matrix(hard=True)[None, None].expand(batch_size, num_graph_samples, -1, -1)
        # adj_matrix: shape [batch_size, num_graph_samples, causal_vars (parents), causal_vars (children)]

        """
        The general idea is to run the MLP for every latent variable for every possible causal assignment of it.
        The sample assignment of target_samples is only used for inputs.
        """
        # Map the causal -> causal adjacency matrix to latent -> causal
        latent_to_causal = (target_samples[:, None, :, :, None] * adj_matrix[:, :, None, :, :]).sum(dim=-2)
        # latent_to_causal: shape [batch, num_graph_samples, latent_vars, causal_vars]

        # Transpose for mask because adj[i,j] means that i->j
        latent_mask = latent_to_causal.transpose(-2, -1)
        # latent_mask: shape [batch, num_graph_samples, causal_vars, latent_vars]

        # Expand mask to run MLP for every latent variable for every possible causal assignment
        # Make sure that no latent variable sees itself
        latent_eye_matrix = torch.eye(latent_mask.shape[-1], device=latent_mask.device)
        latent_mask = latent_mask[:, :, :, None, :] * (1 - latent_eye_matrix)[None, None, None, :, :]
        latent_mask = latent_mask[:, None].expand(-1, num_z_samples, -1, -1, -1, -1)
        # latent_mask: shape [batch, num_z_samples, num_graph_samples, causal_vars, latent_vars (MLP for each latent), latent_vars (input to NN)]
        
        # Mask out inputs for those causal variables that have been intervened upon
        latent_mask = latent_mask * (1 - target_oh[:, None, None, :, None, None])

        if self.autoregressive:
            # Add autoregressive-style variables
            ones_tril = target_samples.new_ones(self.num_latents, self.num_latents).tril(diagonal=-1)
            extra_mask = target_samples.transpose(1, 2)[:, None, None, :, None, :] * ones_tril[None, None, None, None, :, :]
            latent_mask = latent_mask + extra_mask
        
        # Prepare targets as input to the MLPs (-1 for masked elements)
        target_eye_matrix = torch.eye(target_oh.shape[-1], device=latent_mask.device)[None, None, :, None, :]
        target_input = target_oh[:, None, None, None, None, :].expand(-1, num_z_samples, num_graph_samples, -1, self.num_latents, -1) * target_eye_matrix # - (1 - target_eye_matrix)
        # target_input: shape [batch, num_z_samples, num_graph_samples, causal_vars, latent_vars, num_targets]

        # Prepared shared inputs if any
        if z_shared is None:
            z_shared = []
        else:
            # z_shared: shape [batch, num_z_samples, shared_inputs] -> [batch, num_z_samples, num_graph_samples, causal_vars, latent_vars, shared_inputs]
            z_shared = z_shared[:, :, None, None, None, :].expand(-1, -1, num_graph_samples, self.num_blocks, self.num_latents, -1)
            z_shared = [z_shared]

        # Obtain predictions from network
        z_inp = torch.cat([z_sample[:, :, None, None, None, :] * latent_mask, latent_mask * 2 - 1, target_input] + z_shared, dim=-1)
        s = 1
        # For efficiency, we run 1 sample differentiably for the distribution parameters,
        # and all without gradients since ENCO doesn't need gradients through the networks
        if self.gradient_efficient and self.train_graph and num_graph_samples > 1:
            z_inp_flat = z_inp[:, :, :s].flatten(0, 3)
            preds = self.net(z_inp_flat)
            preds = preds.unflatten(0, (batch_size, num_z_samples, s, self.num_blocks))
            with torch.no_grad():
                z_inp_flat_nograd = z_inp[:, :, s:].detach().flatten(0, 3)
                if self.num_latents <= 16:
                    preds_nograd = self.net(z_inp_flat_nograd)
                else:
                    preds_nograd = torch.cat([self.net(z_part) for z_part in z_inp_flat_nograd.chunk(2, dim=0)], dim=0)
                preds_nograd = preds_nograd.unflatten(0, (batch_size, num_z_samples, num_graph_samples - s, self.num_blocks)).detach()
            preds = torch.cat([preds, preds_nograd], dim=2)
        else:
            z_inp_flat = z_inp.flatten(0, 3)
            preds = self.net(z_inp_flat)
            preds = preds.unflatten(0, (batch_size, num_z_samples, num_graph_samples, self.num_blocks))
        prior_mean, prior_logstd = preds.unbind(dim=-1)
        # prior_mean: shape [batch, num_graph_sampels, causal_vars, latent_vars]

        # Calculate KL divergence or Gaussian Log Prob if we have samples
        if z_mean is not None and z_logstd is not None:
            if len(z_mean.shape) == 2:
                z_mean = z_mean[:, None]
                z_logstd = z_logstd[:, None]
            kld = kl_divergence(z_mean[:, :, None, None], z_logstd[:, :, None, None], prior_mean, prior_logstd).mean(dim=1)
        else:
            kld = -gaussian_log_prob(prior_mean[:, None], prior_logstd[:, None], z_sample[:, :, None, None, None, :])
            kld = kld.mean(dim=[1,2])  # Mean samples over samples
        # kld: shape [batch, num_graph_samples, causal_vars, latent_vars]
        
        # VAE KLD
        if not self.gradient_efficient:
            kld_vae = kld.mean(dim=1)  # Mean over graph samples
        else:
            kld_vae = kld[:, :s].mean(dim=1)
        # kld_vae: shape [batch, causal_vars, latent_vars]
        kld_vae = (kld_vae * (target_probs.transpose(0, 1)[None] + 1e-4)).sum(dim=1)  # Weighted mean over causal vars
        kld_vae = kld_vae.sum(dim=-1)  # Sum over latents
        # kld_vae: shape [batch]

        if self.training and self.train_graph:
            if self.graph_learning_method == 'ENCO':
                # ENCO gradients
                with torch.no_grad():
                    # Get theta and gamma probabilities for gradients
                    orient_probs = torch.sigmoid(self.enco_theta)
                    gamma_probs = torch.sigmoid(self.enco_gamma)

                    if len(adj_matrix.shape) == 3:  # Expand adj matrix by batch size if not sampled with it
                        adj_matrix = adj_matrix[None].expand(batch_size, -1, -1, -1)
                    # The NLL of causal var is the sum of all latent vars weighted by their probability to belong
                    # to the corresponding causal variable
                    kld_causal = (kld[:, :, :, :] * target_probs.transpose(0, 1)[None, None]).sum(dim=-1)  # Sum over latents
                    self.kld_per_causal = kld_causal.detach()
                    kld_causal = kld_causal[:, :, None, :]  # Expand since we only consider children NLL

                    # Standard ENCO gradients. Note that we take the positive and negative average per batch element
                    num_pos = adj_matrix.sum(dim=1)
                    num_neg = adj_matrix.shape[1] - num_pos
                    mask = ((num_pos > 0) * (num_neg > 0)).float()
                    pos_grads = (kld_causal * adj_matrix).sum(dim=1) / num_pos.clamp_(min=1e-5)
                    neg_grads = (kld_causal * (1 - adj_matrix)).sum(dim=1) / num_neg.clamp_(min=1e-5)
                    gamma_grads = mask * gamma_probs * (1 - gamma_probs) * orient_probs * (pos_grads - neg_grads + self.lambda_sparse)
                    theta_grads = mask * orient_probs * (1 - orient_probs) * gamma_probs * (pos_grads - neg_grads)

                    gamma_grads = gamma_grads * (1 - target_oh[:,None,:])  # Ignore gradients for intervened vars
                    gamma_grads[:,torch.arange(gamma_grads.shape[1]), torch.arange(gamma_grads.shape[2])] = 0.
                    
                    theta_grads = theta_grads * target_oh[:, :, None]  # Only gradients for intervened vars
                    theta_grads = theta_grads * (1 - target_oh[:, :, None] * target_oh[:, None, :])  # Mask out intervened to intervened
                    theta_grads = theta_grads - theta_grads.transpose(1, 2)  # theta_ij = -theta_ji, and implicitly theta_ii=0

                    self.gamma_grads = gamma_grads.mean(dim=0).detach()
                    self.theta_grads = theta_grads.mean(dim=0).detach()
                # Hook gradients to ENCO parameters on the backward pass
                if kld_vae.requires_grad:
                    kld_vae.register_hook(lambda *args, **kwargs: update_enco_params(self, *args, **kwargs))
            elif self.graph_learning_method == 'NOTEARS':
                # Gradients already through the Gumbel Softmax
                edge_probs = torch.sigmoid(self.notears_params)
                sparsity_regularizer = edge_probs.sum() * self.lambda_sparse
                matrix_exponential = torch.trace(torch.matrix_exp(edge_probs)) - edge_probs.shape[0]
                kld_vae = kld_vae + sparsity_regularizer + matrix_exponential * matrix_exp_factor
        else:
            if self.graph_learning_method == 'ENCO':
                self.gamma_grads = None
                self.theta_grads = None

        return kld_vae


    def logging(self, logger):
        if self.graph_learning_method == 'ENCO':
            for i in range(self.enco_theta.shape[0]):
                for j in range(self.enco_theta.shape[1]):
                    if i == j:
                        continue
                    logger.log(f'enco_theta_{i}_to_{j}', self.enco_theta[i,j], on_step=False, on_epoch=True)
                    logger.log(f'enco_gamma_{i}_to_{j}', self.enco_gamma[i,j], on_step=False, on_epoch=True)
                    if self.enco_theta.grad is not None:
                        logger.log(f'enco_theta_{i}_to_{j}_grads', self.enco_theta.grad[i,j], on_step=False, on_epoch=True)
                    if self.enco_gamma.grad is not None:
                        logger.log(f'enco_gamma_{i}_to_{j}_grads', self.enco_gamma.grad[i,j], on_step=False, on_epoch=True)
        elif self.graph_learning_method == 'NOTEARS':
            for i in range(self.notears_params.shape[0]):
                for j in range(self.notears_params.shape[1]):
                    if i == j:
                        continue
                    logger.log(f'notears_{i}_to_{j}', self.notears_params[i, j], on_step=False, on_epoch=True)
        soft_adj = self.get_adj_matrix(hard=False)
        logger.log('matrix_exponential', torch.trace(torch.matrix_exp(soft_adj)))
        max_grad = (soft_adj * (1 - soft_adj)).max()
        logger.log('adj_matrix_max_grad', max_grad)

    def check_trainability(self):
        soft_adj = self.get_adj_matrix(hard=False)
        max_grad = (soft_adj * (1 - soft_adj)).max().item()
        if max_grad < 1e-3 and self.num_graph_samples > 1:
            print('Freezing graph due to minimal gradients')
            self.num_graph_samples = 1
            self.train_graph = False

    def get_adj_matrix(self, hard=False, for_training=False):
        if hard or (for_training and not self.train_graph):
            if self.graph_learning_method == 'ENCO':
                adj_matrix = torch.logical_and(self.enco_theta > 0.0, self.enco_gamma > 0.0).float()
            elif self.graph_learning_method == 'NOTEARS':
                adj_matrix = (self.notears_params > 0.0).float()
        else:
            if self.graph_learning_method == 'ENCO':
                adj_matrix = torch.sigmoid(self.enco_theta) * torch.sigmoid(self.enco_gamma)
            elif self.graph_learning_method == 'NOTEARS':
                adj_matrix = torch.sigmoid(self.notears_params)
        return adj_matrix.detach()

    def get_target_assignment(self, hard=False):
        # Returns psi, either 'hard' (one-hot, e.g. for triplet eval) or 'soft' (probabilities, e.g. for debug)
        if not hard:
            return torch.softmax(self.target_params, dim=-1)
        else:
            return F.one_hot(torch.argmax(self.target_params, dim=-1), num_classes=self.target_params.shape[-1])


def update_enco_params(self, *args, **kwargs):
    if self.gamma_grads is not None:
        self.enco_gamma.grad = self.gamma_grads.detach()
    if self.theta_grads is not None:
        self.enco_theta.grad = self.theta_grads.detach()