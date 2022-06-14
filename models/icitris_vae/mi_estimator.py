import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

import sys
sys.path.append('../../')
from models.shared import MultivarLinear


class MIEstimator(nn.Module):
    """
    The MI estimator for guiding towards better disentanglement of the causal variables in latent space.
    """

    def __init__(self, num_latents, 
                       c_hid, 
                       num_blocks, 
                       momentum_model=0.0, 
                       var_names=None, 
                       num_layers=1, 
                       act_fn=lambda: nn.SiLU(),
                       gumbel_temperature=1.0,
                       use_normalization=True):
        """
        Parameters
        ----------
        num_latents : int
                      Number of latents of the encoder.
        c_hid : int
                Hidden dimensions to use in the target classifier network
        num_blocks : int
                     Number of blocks to group the latent dimensions into. In other words,
                     it is the number of causal variables plus 1 (psi(0) - unintervened information).
        momentum_model : float
                         During training, we need to run the target classifier once without 
                         gradients, which we implement by using a separate, frozen network. 
                         Following common practice in GANs, we also consider using momentum for 
                         the frozen parameters here. In general, we did not experience it to 
                         have a considerable effect on the performance.
        var_names : list
                    List of names of the causal variables. Used for logging.
        num_layers : int
                     Number of layers to use in the network.
        act_fn : function returning nn.Module
                 Activation function to use in the network.
        gumbel_temperature : float
                             Temperature to use for the Gumbel Softmax sampling.
        use_normalization : bool
                            Whether to use LayerNorm in the target classifier or not.
        """
        super().__init__()
        self.momentum_model = momentum_model
        self.gumbel_temperature = gumbel_temperature
        self.num_comparisons = 1
        self.num_blocks = num_blocks
        self.num_latents = num_latents
        self.c_hid = c_hid * 2

        # Network creation
        self.mi_estimator = nn.Sequential(
                nn.Linear(num_latents * 3 + num_blocks, self.c_hid),
                nn.LayerNorm(self.c_hid),
                act_fn(),
                nn.Linear(self.c_hid, self.c_hid),
                nn.LayerNorm(self.c_hid),
                act_fn(),
                nn.Linear(self.c_hid, 1, bias=False)
            )
        self.mi_estimator[-1].weight.data.fill_(0.0)
        self.exp_mi_estimator = deepcopy(self.mi_estimator)
        for p in self.exp_mi_estimator.parameters():
            p.requires_grad_(False)
        
        # Variable names for logging
        self.var_names = var_names
        if self.var_names is not None:
            if len(self.var_names) <= num_blocks:
                self.var_names = self.var_names + ['No variable']
            if len(self.var_names) <= num_blocks + 1:
                self.var_names = self.var_names + ['All variables']

    @torch.no_grad()
    def _step_exp_avg(self):
        # Update frozen model with momentum on new params
        for p1, p2 in zip(self.mi_estimator.parameters(), self.exp_mi_estimator.parameters()):
            p2.data.mul_(self.momentum_model).add_(p1.data * (1 - self.momentum_model)) 

    def _tag_to_str(self, t):
        # Helper function for finding correct logging names for causal variable indices
        if self.var_names is None or len(self.var_names) <= t:
            return str(t)
        else:
            return f'[{self.var_names[t]}]'

    def forward(self, z_sample, target, transition_prior, logger=None, instant_prob=None):
        """
        Calculates the loss for the mutual information estimator.

        Parameters
        ----------
        z_sample : torch.FloatTensor, shape [batch_size, time_steps, num_latents]
                   The sets of latents for which the loss should be calculated. If time steps is 2, we 
                   use z^t=z_sample[:,0], z^t+1=z_sample[:,1]. If time steps is larger than 2, we apply
                   it for every possible pair over time.
        target : torch.FloatTensor, shape [batch_size, time_steps-1, num_blocks]
                 The intervention targets I^t+1
        transition_prior : TransitionPrior
                           The transition prior of the model. Needed for obtaining the parameters of psi.
        """
        if self.training:
            self._step_exp_avg()

        target = target.flatten(0, 1)
        z_sample_0 = z_sample[:,:-1].flatten(0, 1)
        z_sample_1 = z_sample[:,1:].flatten(0, 1)

        # Find samples for which certain variables have been intervened upon
        with torch.no_grad():
            idxs = [torch.where(target[:,i] == 1)[0] for i in range(self.num_blocks)]
            idxs_stack = torch.cat(idxs, dim=0)
            batch_size = idxs_stack.shape[0]
            i_batch_sizes = [dx.shape[0] for dx in idxs]
            i_batch_sizes_cumsum = np.cumsum(i_batch_sizes)
            intv_target = torch.zeros_like(idxs_stack)
            for i in range(1, self.num_blocks):
                intv_target[i_batch_sizes_cumsum[i-1]:i_batch_sizes_cumsum[i]] = i
            intv_target_onehot = F.one_hot(intv_target, num_classes=self.num_blocks)

        # Sample graphs and latent->causal assignments
        target_assignment = F.gumbel_softmax(transition_prior.target_params[None].expand(batch_size, -1, -1), 
                                             tau=self.gumbel_temperature, hard=True)
        graph_probs = transition_prior.get_adj_matrix(hard=False, for_training=True)
        graph_samples = torch.bernoulli(graph_probs[None].expand(batch_size, -1, -1))
        if instant_prob is not None:  # Mask out instant parents with probability
            graph_mask = (torch.rand(batch_size, graph_probs.shape[1], device=graph_samples.device) < instant_prob).float()
            graph_samples = graph_samples * graph_mask[:,None,:]
        graph_samples = graph_samples - torch.eye(graph_samples.shape[1], device=graph_samples.device)[None]  # Self-connection (-1), parents (1), others (0)
        latent_mask = (target_assignment[:,:,:,None] * graph_samples[:,None,:,:]).sum(dim=-2).transpose(1, 2)

        # Prepare positive pairs
        z_sample_sel_0 = z_sample_0[idxs_stack]
        z_sample_sel_1 = z_sample_1[idxs_stack]
        latent_mask_sel = latent_mask[torch.arange(intv_target.shape[0], dtype=torch.long), intv_target]
        latent_mask_sel_abs = latent_mask_sel.abs()
        inp_sel = torch.cat([z_sample_sel_0, 
                             z_sample_sel_1 * latent_mask_sel_abs, 
                             latent_mask_sel, 
                             intv_target_onehot], dim=-1)

        # Prepare negative pairs
        inp_alts = []
        for _ in range(self.num_comparisons):
            alter_idxs = torch.cat([torch.randperm(i_batch_sizes[i], device=idxs_stack.device) + (0 if i == 0 else i_batch_sizes_cumsum[i-1]) for i in range(self.num_blocks)], dim=0)
            z_sample_alt_1 = z_sample_sel_1[alter_idxs]
            inp_alt = torch.cat([z_sample_sel_0, 
                                 torch.where(latent_mask_sel == -1, z_sample_alt_1, z_sample_sel_1) * latent_mask_sel_abs, 
                                 latent_mask_sel, 
                                 intv_target_onehot], dim=-1)
            inp_alts.append(inp_alt)
        joint_inp = torch.stack([inp_sel] + inp_alts, dim=1)

        # Binary classifier as mutual information estimator
        model_out = self.mi_estimator(joint_inp.detach()).squeeze(dim=-1)
        z_out = self.exp_mi_estimator(joint_inp).squeeze(dim=-1)
        loss_model = -model_out[:,0] + torch.logsumexp(model_out, dim=1)  # Same to -F.log_softmax(z_out, dim=1)[:,0]
        loss_z = -F.log_softmax(z_out, dim=1)[:,-1]  # Alternative is mean over last dimension

        # Finalize loss
        loss_model = loss_model.mean()
        loss_z = loss_z.mean()
        reg_loss = 0.001 * (model_out ** 2).mean()  # To keep outputs in a reasonable range
        loss_model = loss_model + reg_loss

        # Logging
        if logger is not None:
            with torch.no_grad():
                acc = (z_out.argmax(dim=1) == 0).float()
                for b in range(self.num_blocks):
                    num_elem = intv_target_onehot[:,b].sum().item()
                    if num_elem > 0:
                        acc_b = (acc * intv_target_onehot[:,b]).sum() / num_elem
                        logger.log(f'mi_estimator_accuracy_{self._tag_to_str(b)}', acc_b, on_step=False, on_epoch=True)
                logger.log('mi_estimator_output_square', (model_out ** 2).mean(), on_step=False, on_epoch=True)

        return loss_model, loss_z
