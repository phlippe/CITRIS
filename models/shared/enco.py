import torch
import torch.nn as nn
import pytorch_lightning as pl
from copy import deepcopy
from tqdm.auto import tqdm

import sys
sys.path.append('../../')
from models.shared.modules import CosineWarmupScheduler, MultivarLinear
from models.shared.utils import gaussian_log_prob


class ENCOGraphLearning:
    """ Implementation of the causal discovery method "ENCO" for post-processing on a model """

    def __init__(self, model, 
                       default_gamma=0.0, 
                       verbose=True, 
                       num_graph_samples=50, 
                       lambda_sparse=0.01,
                       debug=False):
        self.model = model
        self.debug = debug
        self.verbose = verbose
        self.num_graph_samples = num_graph_samples
        self.lambda_sparse = lambda_sparse
        if hasattr(self.model, 'prior') and hasattr(self.model.prior, 'num_blocks'):
            self.num_blocks = self.model.prior.num_blocks
            self.num_latents = self.model.prior.num_latents
        else:
            self.num_blocks = self.model.hparams.num_causal_vars
            self.num_latents = self.model.hparams.num_latents

        # Check whether the distribution within a causal variable is autoregressive
        self.autoregressive = False
        if (hasattr(self.model, 'prior') and hasattr(self.model.prior, 'autoregressive') and self.model.prior.autoregressive) or \
           (hasattr(self.model.hparams, 'autoregressive_prior') and self.model.hparams.autoregressive_prior):
            self.autoregressive = True
        print(f'ENCO - Autoregressive? {self.autoregressive}')

        if self.autoregressive:
            c_hid = self.model.hparams.c_hid
            self.net = nn.Sequential(
                    MultivarLinear(self.num_latents * 4, c_hid, [self.num_latents]),
                    nn.SiLU(),
                    MultivarLinear(c_hid, c_hid, [self.num_latents]),
                    nn.SiLU(),
                    MultivarLinear(c_hid, 2, 
                                   [self.num_latents])
                ).to(self.model.device)
        else:
            c_hid = self.model.hparams.c_hid * 2
            self.net = nn.Sequential(
                    MultivarLinear(self.num_latents * 4, c_hid, [self.num_blocks]),
                    nn.SiLU(),
                    MultivarLinear(c_hid, c_hid, [self.num_blocks]),
                    nn.SiLU(),
                    MultivarLinear(c_hid, 2*self.num_latents, 
                                   [self.num_blocks])
                ).to(self.model.device)

        # Initialize adjacency matix. If we have already learned an adjacency matrix during training,
        # reuse the orientations found
        if hasattr(self.model, 'prior') and hasattr(self.model.prior, 'get_adj_matrix'):
            prior_adj_matrix = self.model.prior.get_adj_matrix(hard=True)
            self.gamma = nn.Parameter(torch.cat([torch.ones(self.num_blocks, self.num_blocks, device=prior_adj_matrix.device).fill_(default_gamma),
                                                 prior_adj_matrix * default_gamma + (1 - prior_adj_matrix) * -9e15], dim=0))
            self.theta = nn.Parameter((prior_adj_matrix * 2 - 1) * 9e15)
        else:
            self.gamma = nn.Parameter(torch.cat([torch.ones(self.num_blocks, self.num_blocks, device=self.model.device).fill_(default_gamma),
                                                 torch.eye(self.num_blocks, device=self.model.device) * -9e15], dim=0))
            self.theta = nn.Parameter(torch.eye(self.num_blocks, device=self.model.device) * -9e15)

        if hasattr(self.model, 'prior') and hasattr(self.model.prior, 'get_target_assignment'):
            self.target_assignment = self.model.prior.get_target_assignment(hard=True)
        elif hasattr(self.model, 'last_target_assignment'):
            self.target_assignment = self.model.last_target_assignment
        elif hasattr(self.model, 'prior_t1'):
            self.target_assignment = self.model.prior_t1.get_target_assignment(hard=True)[:,:-1]
        else:
            assert False, 'Cannot find target assignment in model.'
        assert self.target_assignment.shape[-1] == self.num_blocks

        if self.autoregressive:
            # Autoregressive mask for instantaneous vars
            ones_tril = self.target_assignment.new_ones(self.num_latents, self.num_latents).tril(diagonal=-1)
            self.autoregressive_mask = (self.target_assignment.transpose(0, 1)[:, None, :] * ones_tril[None, :, :]).sum(dim=0)  # Sum over causal vars
        
        self.model_optimizer = torch.optim.AdamW(self.net.parameters(), lr=2e-3, weight_decay=1e-4)
        self.model_scheduler = CosineWarmupScheduler(self.model_optimizer, warmup=100, max_iters=int(1e7))
        self.gamma_optimizer = torch.optim.Adam([self.gamma], lr=5e-3, betas=(0.9, 0.9))
        self.theta_optimizer = torch.optim.Adam([self.theta], lr=1e-2, betas=(0.9, 0.999))

        self.latents_means = torch.zeros(self.num_latents, device=self.gamma.device)
        self.latents_stds = torch.ones(self.num_latents, device=self.gamma.device)
        self.gamma_log = []

    def iterator(self, it, desc=None, leave=False):
        if self.verbose:
            return tqdm(it, desc=desc, leave=leave)
        else:
            return it

    def learn_graph(self, dataset, num_epochs=40):
        self.model.eval()
        self.model.freeze()
        with torch.enable_grad():
            dist_data_loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
            graph_data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
            self.prepare_latent_statistics(dist_data_loader)
            for _ in self.iterator(range(10 if not self.debug else 1), 'Distribution pretraining'): # First fit distributions well
                self.distribution_fitting_epoch(dist_data_loader)
            for _ in self.iterator(range(num_epochs), desc='ENCO epochs'):
                if self.is_gamma_saturated():
                    continue
                self.distribution_fitting_epoch(dist_data_loader)
                self.graph_fitting_epoch(graph_data_loader)
            return self.get_adj_matrix()

    @torch.no_grad()
    def prepare_latent_statistics(self, data_loader):
        encodings = []
        for latents, _ in self.iterator(data_loader, 'Latent statistics'):
            latents = latents.to(self.target_assignment.device)[:,0]
            encodings.append(self.model.encode(latents))
        encodings = torch.cat(encodings, dim=0)
        self.latents_means = encodings.mean(dim=0).detach()
        self.latents_stds = encodings.std(dim=0).detach()

    def distribution_fitting_epoch(self, data_loader, max_steps=1000):
        """
        Learn the conditional distributions of the causal variables
        """
        self.net.train()
        if self.debug:
            max_steps = 10
        data_iter = iter(data_loader)
        for _ in self.iterator(range(min(max_steps, len(data_iter))), 'Distribution fitting'):
            latents, targets = next(data_iter)
            latents = latents.to(self.target_assignment.device)
            targets = targets.to(self.target_assignment.device)
            latents = self.encode_latent_batch(latents)
            causal_graphs = self.sample_graphs(latents.shape[0])
            nll = self.run_priors(latents, causal_graphs)
            nll = nll * (1 - targets)
            loss = nll.mean()
            self.model_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 0.1, error_if_nonfinite=True)
            self.model_optimizer.step()
            self.model_scheduler.step()

    def sample_graphs(self, batch_size):
        gamma_sigm = torch.sigmoid(self.gamma.detach())
        theta_sigm = torch.sigmoid(self.theta.detach())
        gamma_sigm[self.num_blocks:] *= theta_sigm
        causal_graphs = torch.bernoulli(gamma_sigm[None].expand(batch_size, -1, -1))
        return causal_graphs

    def graph_fitting_epoch(self, data_loader, max_steps=100):
        """
        Fit graph parameters with the learned distributions
        """
        self.net.eval()
        if self.debug:
            max_steps = 10
        data_iter = iter(data_loader)
        for _ in self.iterator(range(min(max_steps, len(data_iter))), 'Graph fitting'):
            latents, targets = next(data_iter)
            with torch.no_grad():
                latents = latents.to(self.target_assignment.device)
                targets = targets.to(self.target_assignment.device)
                latents = self.encode_latent_batch(latents)
                causal_graphs = self.sample_graphs(self.num_graph_samples)
                causal_graphs_exp = causal_graphs[None].expand(latents.shape[0], -1, -1, -1).flatten(0, 1)
                latents_exp = latents[:,None].expand(-1, self.num_graph_samples, -1, -1).flatten(0, 1)
                nll = self.run_priors(latents_exp, causal_graphs_exp)
                nll = nll.unflatten(0, (-1, self.num_graph_samples))
                
                causal_graphs_exp = causal_graphs_exp.unflatten(0, (latents.shape[0], self.num_graph_samples))
                gamma_sigm = torch.sigmoid(self.gamma.detach())
                theta_sigm = torch.sigmoid(self.theta.detach())
                targets = targets.squeeze(dim=1)  # Shape [batch, num_causal_vars]
                num_pos = causal_graphs_exp.sum(dim=1)
                num_neg = self.num_graph_samples - num_pos
                mask = ((num_pos > 0) * (num_neg > 0)).float()

                pos_grads = (nll[:, :, None] * causal_graphs_exp).sum(dim=1) / num_pos.clamp_(min=1e-5)
                neg_grads = (nll[:, :, None] * (1 - causal_graphs_exp)).sum(dim=1) / num_neg.clamp_(min=1e-5)
                gamma_grads = mask * gamma_sigm * (1 - gamma_sigm) * (pos_grads - neg_grads + self.lambda_sparse)
                gamma_grads[:, self.num_blocks:] *= theta_sigm
                gamma_grads = gamma_grads * (1 - targets[:, None, :])  # Targets shape [Batch, 1, num_causal_vars]                
                gamma_grads[:,self.num_blocks + torch.arange(gamma_grads.shape[2]), torch.arange(gamma_grads.shape[2])] = 0.

                theta_grads = theta_sigm * (1 - theta_sigm) * (mask * gamma_sigm * (pos_grads - neg_grads))[:, self.num_blocks:]
                theta_grads = theta_grads * targets[:, :, None]  # Only gradients for intervened vars
                theta_grads = theta_grads * (1 - targets[:, :, None] * targets[:, None, :])  # Mask out intervened to intervened
                theta_grads = theta_grads - theta_grads.transpose(1, 2)  # theta_ij = -theta_ji, and implicitly theta_ii=0

                gamma_grads = gamma_grads.mean(dim=0)
                theta_grads = theta_grads.mean(dim=0)

            self.gamma_optimizer.zero_grad()
            self.theta_optimizer.zero_grad()
            self.gamma.grad = gamma_grads
            self.theta.grad = theta_grads
            self.gamma_optimizer.step()
            self.theta_optimizer.step()
        self.gamma_log.append(self.gamma.data.detach().clone())

    @torch.no_grad()
    def encode_latent_batch(self, latents):
        latents = self.model.encode(latents.flatten(0, 1))
        latents = (latents - self.latents_means[None]) / self.latents_stds[None]
        latents = latents.unflatten(0, (-1, 2))
        return latents.detach()

    def run_priors(self, latents, causal_graphs):
        # Given a set of latents and causal graphs, return nll per causal variable
        z0 = latents[:, 0]
        z1 = latents[:, 1]

        if self.autoregressive:
            latent_mask0 = (self.target_assignment[None, :, :, None] * causal_graphs[:, None, :self.num_blocks, :]).sum(dim=-2)
            latent_mask0 = (self.target_assignment[None, None, :, :] * latent_mask0[:, :, None, :]).sum(dim=-1)
            latent_mask0 = latent_mask0.transpose(-2, -1) # i -> j => Transpose to j <- i
            latent_mask1 = (self.target_assignment[None, :, :, None] * causal_graphs[:, None, self.num_blocks:, :]).sum(dim=-2)
            latent_mask1 = (self.target_assignment[None, None, :, :] * latent_mask1[:, :, None, :]).sum(dim=-1)
            latent_mask1 = latent_mask1.transpose(-2, -1) # i -> j => Transpose to j <- i
            latent_mask1 = latent_mask1 + self.autoregressive_mask[None, :, :]

            inp = torch.cat([z0[:, None] * latent_mask0, z1[:, None] * latent_mask1, latent_mask0 * 2 - 1, latent_mask1 * 2 - 1], dim=-1)

            prior_mean, prior_logstd = self.net(inp).unbind(dim=-1)
            nll = -gaussian_log_prob(prior_mean, prior_logstd, z1)
            nll = (nll[:,:,None] * self.target_assignment[None]).sum(dim=1)
        else:
            latent_mask0 = (self.target_assignment[None, :, :, None] * causal_graphs[:, None, :self.num_blocks, :]).sum(dim=-2)
            latent_mask0 = latent_mask0.transpose(-2, -1) # i -> j => Transpose to j <- i
            latent_mask1 = (self.target_assignment[None, :, :, None] * causal_graphs[:, None, self.num_blocks:, :]).sum(dim=-2)
            latent_mask1 = latent_mask1.transpose(-2, -1) # i -> j => Transpose to j <- i
            
            inp = torch.cat([z0[:, None] * latent_mask0, z1[:, None] * latent_mask1, latent_mask0 * 2 - 1, latent_mask1 * 2 - 1], dim=-1)

            prior_mean, prior_logstd = self.net(inp).chunk(2, dim=-1)
            nll = -gaussian_log_prob(prior_mean, prior_logstd, z1[:,None])
            nll = (nll * self.target_assignment.transpose(0, 1)[None]).sum(dim=-1)
        return nll

    def get_adj_matrix(self):
        adj_matrix = (self.gamma.data > 0.0).long().detach().cpu()
        temporal_adj_matrix = adj_matrix[:self.num_blocks]
        instantaneous_adj_matrix = adj_matrix[self.num_blocks:] * (self.theta.data > 0.0).long().detach().cpu()
        return temporal_adj_matrix, instantaneous_adj_matrix

    def is_gamma_saturated(self):
        gamma_sigm = torch.sigmoid(self.gamma)
        max_grad = (gamma_sigm * (1 - gamma_sigm)).max()
        return max_grad.item() < 1e-3

    def get_instantaneous_parameters(self):
        return torch.sigmoid(self.gamma).detach()[self.num_blocks:], torch.sigmoid(self.theta).detach()
