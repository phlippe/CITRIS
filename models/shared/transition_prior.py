import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

import sys
sys.path.append('../../')
from models.shared.utils import kl_divergence, gaussian_log_prob
from models.shared.modules import MultivarLinear, AutoregLinear



class AutoregressiveConditionalPrior(nn.Module):
    """
    The autoregressive base model for the autoregressive transition prior.
    The model is inspired by MADE and uses linear layers with masked weights.
    """

    def __init__(self, num_latents, num_blocks, c_hid, c_out, imperfect_interventions=True):
        """
        Parameters
        ----------
        num_latents : int
                      Number of latent dimensions.
        num_blocks : int
                     Number of blocks to group the latent dimensions into. In other words,
                     it is the number of causal variables plus 1 (psi(0) - unintervened information).
        c_hid : int
                Hidden dimensionality to use in the network.
        c_out : int
                Output dimensionality per latent dimension (2 for Gaussian param estimation)
        imperfect_interventions : bool
                                  Whether interventions may be imperfect or not. If not, we mask the
                                  conditional information on interventions for a slightly faster
                                  convergence.
        """
        super().__init__()
        # Input layer for z_t
        self.context_layer = nn.Linear(num_latents, num_latents * c_hid)
        # Input layer for I_t
        self.target_layer = nn.Linear(num_blocks, num_latents * c_hid)
        # Autoregressive input layer for z_t+1
        self.init_layer = AutoregLinear(num_latents, 2, c_hid, diagonal=False)
        # Autoregressive main network with masked linear layers
        self.net = nn.Sequential(
                nn.SiLU(),
                AutoregLinear(num_latents, c_hid, c_hid, diagonal=True),
                nn.SiLU(),
                AutoregLinear(num_latents, c_hid, c_out, diagonal=True)
            )
        self.num_latents = num_latents
        self.imperfect_interventions = imperfect_interventions
        self.register_buffer('target_mask', torch.eye(num_blocks))

    def forward(self, z_samples, z_previous, target_samples, target_true):
        """
        Given latent variables z^t+1, z^t, intervention targets I^t+1, and 
        causal variable assignment samples from psi, estimate the prior 
        parameters of p(z^t+1|z^t, I^t+1). This is done by running the
        autoregressive prior for each causal variable to estimate 
        p(z_psi(i)^t+1|z^t, I_i^t+1), and stacking the i-dimension.


        Parameters
        ----------
        z_samples : torch.FloatTensor, shape [batch_size, num_latents]
                    The values of the latent variables at time step t+1, i.e. z^t+1.
        z_previous : torch.FloatTensor, shape [batch_size, num_latents]
                     The values of the latent variables at time step t, i.e. z^t.
        target_samples : torch.FloatTensor, shape [batch_size, num_latents, num_blocks]
                         The sampled one-hot vectors of psi for assigning latent variables to 
                         causal variables.
        target_true : torch.FloatTensor, shape [batch_size, num_blocks]
                      The intervention target vector I^t+1.
        """
        target_samples = target_samples.permute(0, 2, 1)  # shape: [batch_size, num_blocks, num_latents]
        
        # Transform z^t into a feature vector. Expand over number of causal variables to run the prior i-times.
        context_feats = self.context_layer(z_previous)
        context_feats = context_feats.unsqueeze(dim=1)  # shape: [batch_size, 1, num_latents * c_hid]

        # Transform I^t+1 into feature vector, where only the i-th element is shown to the respective masked split.
        target_inp = target_true[:,None] * self.target_mask[None]  # shape: [batch_size, num_blocks, num_latents] 
        target_inp = target_inp - (1 - self.target_mask[None]) # Set -1 for masked values
        target_feats = self.target_layer(target_inp)

        # Mask z^t+1 according to psi samples
        masked_samples = z_samples[:,None] * target_samples # shape: [batch_size, num_blocks, num_latents]
        masked_samples = torch.stack([masked_samples, target_samples*2-1], dim=-1)
        masked_samples = masked_samples.flatten(-2, -1) # shape: [batch_size, num_blocks, 2*num_latents]
        init_feats = self.init_layer(masked_samples)

        if not self.imperfect_interventions:
            # Mask out context features when having perfect interventions
            context_feats = context_feats * (1 - target_true[...,None])

        # Sum all features and use as input to feature network (division by 2 for normalization)
        feats = (target_feats + init_feats + context_feats) / 2.0
        pred_params = self.net(feats)

        # Return prior parameters with first dimension stacking the different causal variables
        pred_params = pred_params.unflatten(-1, (self.num_latents, -1))  # shape: [batch_size, num_blocks, num_latents, c_out]
        return pred_params


class TransitionPrior(nn.Module):
    """
    The full transition prior promoting disentanglement of the latent variables across causal factors.
    """

    def __init__(self, num_latents, num_blocks, c_hid,
                 imperfect_interventions=False,
                 autoregressive_model=False,
                 lambda_reg=0.01,
                 gumbel_temperature=1.0):
        """
        Parameters
        ----------
        num_latents : int
                      Number of latent dimensions.
        num_blocks : int
                     Number of blocks to group the latent dimensions into. In other words,
                     it is the number of causal variables plus 1 (psi(0) - unintervened information).
        c_hid : int
                Hidden dimensionality to use in the prior network.
        imperfect_interventions : bool
                                  Whether interventions may be imperfect or not.
        autoregressive_model : bool
                               If True, an autoregressive prior model is used.
        lambda_reg : float
                     Regularizer for promoting intervention-independent information to be modeled
                     in psi(0)
        gumbel_temperature : float
                             Temperature to use for the Gumbel Softmax sampling.
        """
        super().__init__()
        self.num_latents = num_latents
        self.imperfect_interventions = imperfect_interventions
        self.gumbel_temperature = gumbel_temperature
        self.num_blocks = num_blocks
        self.autoregressive_model = autoregressive_model
        self.lambda_reg = lambda_reg
        assert self.lambda_reg >= 0 and self.lambda_reg < 1.0, 'Lambda regularizer must be between 0 and 1, excluding 1.'

        # Gumbel Softmax parameters of psi. Note that we model psi(0) in the last dimension for simpler implementation
        self.target_params = nn.Parameter(torch.zeros(num_latents, num_blocks + 1))
        if self.lambda_reg <= 0.0:  # No regularizer -> no reason to model psi(0)
            self.target_params.data[:,-1] = -9e15

        # For perfect interventions, we model the prior's parameters under intervention as a simple parameter vector here.
        if not self.imperfect_interventions:
            self.intv_prior = nn.Parameter(torch.zeros(num_latents, num_blocks, 2).uniform_(-0.5, 0.5))
        else:
            self.intv_prior = None

        # Prior model creation
        if autoregressive_model:
            self.prior_model = AutoregressiveConditionalPrior(num_latents, num_blocks+1, 16, 2,
                                                              imperfect_interventions=self.imperfect_interventions)
        else:
            # Simple MLP per latent variable
            self.context_layer = nn.Linear(num_latents, 
                                           c_hid*self.num_latents)
            self.inp_layer = MultivarLinear(1 + (self.num_blocks+1 if self.imperfect_interventions else 0), 
                                            c_hid, [self.num_latents])
            self.out_layer = nn.Sequential(
                    nn.SiLU(),
                    MultivarLinear(c_hid, c_hid, [self.num_latents]),
                    nn.SiLU(),
                    MultivarLinear(c_hid, 2, 
                                   [self.num_latents])
                )

    def _get_prior_params(self, z_t, target=None, target_prod=None, target_samples=None, z_t1=None):
        """
        Abstracting the execution of the networks for estimating the prior parameters.

        Parameters
        ----------
        z_t : torch.FloatTensor, shape [batch_size, num_latents]
              Latents at time step t, i.e. the input to the prior
        target : torch.FloatTensor, shape [batch_size, num_blocks]
                 The intervention targets I^t+1
        target_prod : torch.FloatTensor, shape [batch_size, num_latents, num_blocks]
                      The true targets multiplied with the target sample mask, where masked
                      intervention targets are replaced with -1 to distinguish it from 0s.
        target_samples : torch.FloatTensor, shape [batch_size, num_latents, num_blocks]
                         The sampled one-hot vectors of psi for assigning latent variables to 
                         causal variables.
        z_t1 : torch.FloatTensor, shape [batch_size, num_latents]
               Latents at time step t+1, i.e. the latents for which the prior parameters are estimated.
        """
        if self.autoregressive_model:
            prior_params = self.prior_model(z_samples=z_t1, 
                                            z_previous=z_t, 
                                            target_samples=target_samples,
                                            target_true=target)
            prior_params = prior_params.unbind(dim=-1)
        else:
            net_inp = z_t
            context = self.context_layer(net_inp).unflatten(-1, (self.num_latents, -1))
            net_inp_exp = net_inp.unflatten(-1, (self.num_latents, -1))
            if self.imperfect_interventions:
                if target_prod is None:
                    target_prod = net_inp_exp.new_zeros(net_inp_exp.shape[:-1] + (self.num_blocks,))
                net_inp_exp = torch.cat([net_inp_exp, target_prod], dim=-1)
            block_inp = self.inp_layer(net_inp_exp)
            prior_params = self.out_layer(context + block_inp)
            prior_params = prior_params.chunk(2, dim=-1)
            prior_params = [p.flatten(-2, -1) for p in prior_params]
        return prior_params

    def kl_divergence(self, z_t, target, z_t1_mean, z_t1_logstd, z_t1_sample):
        """
        Calculating the KL divergence between this prior's estimated parameters and
        the encoder on x^t+1 (CITRIS-VAE). Since this prior is in general much more
        computationally cheaper than the encoder/decoder, we marginalize the KL
        divergence over the target assignments for each latent where possible.

        Parameters
        ----------
        z_t : torch.FloatTensor, shape [batch_size, num_latents]
              Latents at time step t, i.e. the input to the prior
        target : torch.FloatTensor, shape [batch_size, num_blocks]
                 The intervention targets I^t+1
        z_t1_mean : torch.FloatTensor, shape [batch_size, num_latents]
                    The mean of the predicted Gaussian encoder(x^t+1)
        z_t1_logstd : torch.FloatTensor, shape [batch_size, num_latents]
                      The log-standard deviation of the predicted Gaussian encoder(x^t+1)
        z_t1_sample : torch.FloatTensor, shape [batch_size, num_latents]
                      A sample from the encoder distribution encoder(x^t+1), i.e. z^t+1
        """
        if len(target.shape) == 1:
            target_oh = F.one_hot(target, num_classes=self.num_blocks)
        else:
            target_oh = target

        # Sample a latent-to-causal assignment from psi
        target_probs = torch.softmax(self.target_params, dim=-1)
        target_samples = F.gumbel_softmax(self.target_params[None].expand(target.shape[0], -1, -1), 
                                       tau=self.gumbel_temperature, hard=True)
        full_target_samples = target_samples
        target_samples, no_target = target_samples[:,:,:-1], target_samples[:,:,-1]
        # Add I_0=0, i.e. no interventions on the noise/intervention-independent variables
        target_exp = torch.cat([target_oh, target_oh.new_zeros(target_oh.shape[0], 1)], dim=-1)

        if self.autoregressive_model:
            # Run autoregressive model
            prior_params = self._get_prior_params(z_t, target_samples=full_target_samples, target=target_exp, z_t1=z_t1_sample)
            kld_all = self._get_kld(z_t1_mean[:,None], z_t1_logstd[:,None], prior_params)
            # Regularize psi(0)
            if self.lambda_reg > 0.0:
                target_probs = torch.cat([target_probs[:,-1:], target_probs[:,:-1] * (1 - self.lambda_reg)], dim=-1)
            # Since to predict the parameters of z^t+1_i, we do not involve whether the target sample of i has been a certain value,
            # we can marginalize it over all possible target samples here.
            kld = (kld_all * target_probs.permute(1, 0)[None]).sum(dim=[1,2])
        elif not self.imperfect_interventions:
            # For perfect interventions, we can estimate p(z^t+1|z^t) and p(z^t+1_j|I^t+1_i=1) independently.
            prior_params = self._get_prior_params(z_t, target_samples=full_target_samples, target=target_exp, z_t1=z_t1_sample)
            kld_std = self._get_kld(z_t1_mean, z_t1_logstd, prior_params)
            intv_params = self._get_intv_params(z_t1_mean.shape, target=None)
            kld_intv = self._get_kld(z_t1_mean[...,None], z_t1_logstd[...,None], intv_params)
        
            target_probs, no_target_probs = target_probs[:,:-1], target_probs[:,-1]
            masked_intv_probs = (target_probs[None] * target_oh[:,None,:])  # shape: [batch_size, num_latents, num_blocks]
            intv_probs = masked_intv_probs.sum(dim=-1)
            no_intv_probs = 1 - intv_probs - no_target_probs
            kld_intv_summed = (kld_intv * masked_intv_probs).sum(dim=-1)  # shape: [batch_size, num_latents]
            # Regularize by multiplying the KLD of psi(0) with 1-lambda_reg
            kld = kld_intv_summed + kld_std * (no_intv_probs + no_target_probs * (1 - self.lambda_reg))
            kld = kld.sum(dim=-1)
        else:
            # For imperfect interventions, we estimate p(z^t+1_i|z^t,I^t+1_j) for all i,j, and marginalize over j.
            net_inps = z_t[:,None].expand(-1, self.num_blocks+1, -1).flatten(0, 1)
            target_samples = torch.eye(self.num_blocks+1, device=z_t.device)[None]
            target_oh = torch.cat([target_oh, target_oh.new_zeros(target_oh.shape[0], 1)], dim=-1)
            target_prod = target_oh[:,None] * target_samples - (1 - target_samples)
            target_prod = target_prod[:,:,None].expand(-1, -1, z_t.shape[-1], -1)
            target_prod = target_prod.flatten(0, 1)
            prior_params = self._get_prior_params(net_inps, target_prod=target_prod)
            prior_params = [p.unflatten(0, (-1, self.num_blocks+1)) for p in prior_params]
            kld = self._get_kld(z_t1_mean[:,None], z_t1_logstd[:,None], prior_params)
            kld = (kld * target_probs.permute(1, 0)[None,:,:])
            kld = kld[...,:-1].sum(dim=[1,2]) + (1 - self.lambda_reg) * kld[...,-1].sum(dim=1)
        
        return kld

    def sample_based_nll(self, z_t, z_t1, target):
        """
        Calculate the negative log likelihood of p(z^t1|z^t,I^t+1), meant for CITRIS-NF.
        We cannot make use of the KL divergence since the normalizing flow transforms
        the autoencoder distribution in a per-sample fashion. Nonetheless, to improve
        stability and smooth gradients, we allow z^t and z^t1 to be multiple samples
        for the same batch elements. 

        Parameters
        ----------
        z_t : torch.FloatTensor, shape [batch_size, num_samples, num_latents]
              Latents at time step t, i.e. the input to the prior, with potentially
              multiple samples over the second dimension.
        z_t1 : torch.FloatTensor, shape [batch_size, num_samples, num_latents]
               Latents at time step t+1, i.e. the samples to estimate the prior for.
               Multiple samples again over the second dimension.
        target : torch.FloatTensor, shape [batch_size, num_blocks]
                 The intervention targets I^t+1
        """
        batch_size, num_samples, _ = z_t.shape

        if len(target.shape) == 1:
            target_oh = F.one_hot(target, num_classes=self.num_blocks)
        else:
            target_oh = target
        
        # Sample a latent-to-causal assignment from psi
        target_probs = torch.softmax(self.target_params, dim=-1)
        target_samples = F.gumbel_softmax(self.target_params[None].expand(batch_size * num_samples, -1, -1), 
                                       tau=self.gumbel_temperature, hard=True)
        # Add sample dimension and I_0=0 to the targets
        target_exp = target_oh[:,None].expand(-1, num_samples, -1).flatten(0, 1)
        target_exp = torch.cat([target_exp, target_exp.new_zeros(batch_size * num_samples, 1)], dim=-1)
        target_prod = target_exp[:,None,:] * target_samples - (1 - target_samples)

        # Obtain estimated prior parameters for p(z^t1|z^t,I^t+1)
        prior_params = self._get_prior_params(z_t.flatten(0, 1), 
                                              target_samples=target_samples, 
                                              target=target_exp,
                                              target_prod=target_prod,
                                              z_t1=z_t1.flatten(0, 1))
        if self.autoregressive_model:
            prior_mean, prior_logstd = [p.unflatten(0, (batch_size, num_samples)) for p in prior_params]
            # prior_mean - shape [batch_size, num_samples, num_blocks, num_latents]
            # prior_logstd - shape [batch_size, num_samples, num_blocks, num_latents]
            # z_t1 - shape [batch_size, num_samples, num_latents]
            z_t1 = z_t1[:,:,None,:]  # Expand by block dimension
            nll = -gaussian_log_prob(prior_mean[:,:,None,:,:], prior_logstd[:,:,None,:,:], z_t1[:,None,:,:,:])
            # We take the mean over samples, both over the z^t and z^t+1 samples.
            nll = nll.mean(dim=[1, 2])  # shape [batch_size, num_blocks, num_latents]
            # Marginalize over target assignment
            nll = nll * target_probs.permute(1, 0)[None]
            nll = nll.sum(dim=[1, 2])  # shape [batch_size]
        elif not self.imperfect_interventions:
            prior_mean, prior_logstd = [p.unflatten(0, (batch_size, num_samples)) for p in prior_params]
            nll_std = -gaussian_log_prob(prior_mean[...,None,:], prior_logstd[...,None,:], z_t1[...,None,:,:])
            nll_std = nll_std.mean(dim=[1, 2])  # Averaging over input and output samples

            intv_params = self._get_intv_params(z_t.shape, target=None)
            intv_mean, intv_logstd = intv_params[0], intv_params[1]
            nll_intv = -gaussian_log_prob(intv_mean[:,None,:,:].detach(), 
                                          intv_logstd[:,None,:,:].detach(), 
                                          z_t1[...,None])
            nll_intv = nll_intv.mean(dim=1)  # Averaging over output samples (input all the same)
            # nll_intv - shape: [batch_size, num_latents, num_blocks]

            target_probs, no_target_probs = target_probs[:,:-1], target_probs[:,-1]
            masked_intv_probs = (target_probs[None] * target_oh[:,None,:])  # shape: [batch_size, num_latents, num_blocks]
            intv_probs = masked_intv_probs.sum(dim=-1)
            no_intv_probs = 1 - intv_probs - no_target_probs * (self.lambda_reg)
            nll_intv_summed = (nll_intv * masked_intv_probs).sum(dim=-1)  # shape: [batch_size, num_latents]
            nll = nll_intv_summed + nll_std * no_intv_probs
            nll = nll.sum(dim=-1)

            nll_intv_noz = -gaussian_log_prob(intv_mean[:,None,:,:], 
                                              intv_logstd[:,None,:,:], 
                                              z_t1[...,None].detach())
            nll_intv_noz = nll_intv_noz.mean(dim=1)
            nll_intv_noz = (nll_intv_noz * target_oh[:,None,:]).sum(dim=[1,2])
            nll = nll + nll_intv_noz - nll_intv_noz.detach()  # No noticable increase in loss, only grads
        else:
            prior_mean, prior_logstd = [p.unflatten(0, (batch_size, num_samples)) for p in prior_params]
            nll = -gaussian_log_prob(prior_mean[...,None,:], prior_logstd[...,None,:], z_t1[...,None,:,:])
            nll = nll.mean(dim=[1, 2])  # Averaging over input and output samples
            nll = nll.sum(dim=-1)

        # In Normalizing Flows, we do not have a KL divergence to easily regularize with lambda_reg.
        # Instead, we regularize the Gumbel Softmax parameters to maximize the probability for psi(0).
        if self.lambda_reg > 0.0 and self.training:
            target_params_soft = torch.softmax(target_params, dim=-1)
            nll = nll + self.lambda_reg * (1-target_params_soft[:,-1]).mean(dim=0)
        
        return nll

    def _get_kld(self, true_mean, true_logstd, prior_params):
        # Function for cleaning up KL divergence calls
        kld = kl_divergence(true_mean, true_logstd, prior_params[0], prior_params[1])
        return kld

    def _get_intv_params(self, shape, target):
        # Return the prior parameters for p(z^t+1_j|I^t+1_i=1)
        intv_params = self.intv_prior[None].expand(shape[0], -1, -1, -1)
        if target is not None:
            intv_params = (intv_params * target[:,None,:,None]).sum(dim=2)
        return intv_params[...,0], intv_params[...,1]

    def get_target_assignment(self, hard=False):
        # Returns psi, either 'hard' (one-hot, e.g. for triplet eval) or 'soft' (probabilities, e.g. for debug)
        if not hard:
            return torch.softmax(self.target_params, dim=-1)
        else:
            return F.one_hot(torch.argmax(self.target_params, dim=-1), num_classes=self.target_params.shape[-1])