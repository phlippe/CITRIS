import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy


class TargetClassifier(nn.Module):
    """
    The target classifier for guiding towards better disentanglement of the causal variables in latent space.
    """

    def __init__(self, num_latents, 
                       c_hid, 
                       num_blocks, 
                       momentum_model=0.0, 
                       var_names=None, 
                       num_layers=1, 
                       act_fn=lambda: nn.SiLU(),
                       gumbel_temperature=1.0,
                       use_normalization=True,
                       use_conditional_targets=False):
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
        use_conditional_targets : bool
                                  If True, we record conditional targets p(I^t+1_i|I^t+1_j).
                                  Needed when intervention targets are confounded.
        """
        super().__init__()
        self.momentum_model = momentum_model
        self.gumbel_temperature = gumbel_temperature
        self.num_blocks = num_blocks
        self.use_conditional_targets = use_conditional_targets
        self.dist_steps = 0.0

        # Network creation
        norm = lambda c: (nn.LayerNorm(c) if use_normalization else nn.Identity())
        layers = [nn.Linear(3*num_latents, 2*c_hid), norm(2*c_hid), act_fn()]
        inp_dim = 2*c_hid
        for _ in range(num_layers - 1):
            layers += [nn.Linear(inp_dim, c_hid), norm(c_hid), act_fn()]
            inp_dim = c_hid
        layers += [nn.Linear(inp_dim, num_blocks)]
        self.classifiers = nn.Sequential(*layers)
        self.classifiers[-1].weight.data.fill_(0.0)
        self.exp_classifiers = deepcopy(self.classifiers)
        for p in self.exp_classifiers.parameters():
            p.requires_grad_(False)

        # Buffers for recording p(I^t+1_i) / p(I^t+1_i|I^t+1_j) in the training data
        self.register_buffer('avg_dist', torch.zeros(num_blocks, 2).fill_(0.5))
        if use_conditional_targets:
            self.register_buffer('avg_cond_dist', torch.zeros(num_blocks, num_blocks, 2, 2).fill_(0.5))
            self.register_buffer('dist_cond_steps', torch.zeros(num_blocks, 2).fill_(1))
        
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
        for p1, p2 in zip(self.classifiers.parameters(), self.exp_classifiers.parameters()):
            p2.data.mul_(self.momentum_model).add_(p1.data * (1 - self.momentum_model)) 

    @torch.no_grad()
    def _update_dist(self, target):
        # Add target tensor to average of target distributions
        if self.dist_steps < 1e6:  # At this time we should have a pretty good average
            target = target.float()
            avg_target = target.mean(dim=[0,1])
            new_dist = torch.stack([1-avg_target, avg_target], dim=-1)
            self.avg_dist.mul_(self.dist_steps/(self.dist_steps + 1)).add_(new_dist * (1./(self.dist_steps + 1)))
            
            if hasattr(self, 'avg_cond_dist'):
                target_sums = target.sum(dim=[0,1])
                target_prod = (target[...,None,:] * target[...,:,None]).sum(dim=[0,1])

                one_cond_one = target_prod / target_sums[None,:].clamp(min=1e-5)
                zero_cond_one = 1 - one_cond_one
                inv_sum = (target.shape[0] * target.shape[1] - target_sums)
                one_cond_zero = (target_sums[:,None] - target_prod) / inv_sum[None,:].clamp(min=1e-5)
                zero_cond_zero = 1 - one_cond_zero
                new_cond_steps = torch.stack([target.shape[0] * target.shape[1] - target_sums, target_sums], dim=-1)
                update_factor = (self.dist_cond_steps/(self.dist_cond_steps + new_cond_steps))[None,:,:,None]
                cond_dist = torch.stack([zero_cond_zero, one_cond_zero, zero_cond_one, one_cond_one], dim=-1).unflatten(-1, (2, 2))
                self.avg_cond_dist.mul_(update_factor).add_(cond_dist * (1 - update_factor))
                self.dist_cond_steps.add_(new_cond_steps)
            self.dist_steps += 1

    def _tag_to_str(self, t):
        # Helper function for finding correct logging names for causal variable indices
        if self.var_names is None or len(self.var_names) <= t:
            return str(t)
        else:
            return f'[{self.var_names[t]}]'

    def forward(self, z_sample, target, transition_prior, logger=None):
        """
        Calculates the loss for the target classifier (predict all intervention targets from all sets of latents)
        and the latents + psi (predict only its respective intervention target from a set of latents).

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
            self._update_dist(target)

        # Joint preparations
        batch_size = z_sample.shape[0]
        time_steps = z_sample.shape[1] - 1
        num_classes = target.shape[-1]
        # Sample latent-to-causal variable assignments
        target_assignment = F.gumbel_softmax(transition_prior.target_params[None].expand(batch_size, time_steps, -1, -1), 
                                             tau=self.gumbel_temperature, hard=True)
        if target_assignment.shape[-1] == num_classes:
            target_assignment = torch.cat([target_assignment, target_assignment.new_zeros(target_assignment.shape[:-1] + (1,))], dim=-1)
        target_assignment = torch.cat([target_assignment, target_assignment.new_ones(target_assignment.shape[:-1] + (1,))], dim=-1)
        num_targets = target_assignment.shape[-1]
        target_assignment = target_assignment.permute(0, 1, 3, 2)
        z_sample = z_sample[...,None,:].expand(-1, -1, num_targets, -1)
        exp_target = target[...,None,:].expand(-1, -1, num_targets, -1).flatten(0, 2).float()
        
        # We consider 2 + [number of causal variables] sets of latents: 
        # (1) one per causal variable, (2) the noise/'no-causal-var' slot psi(0), (3) all latents
        # The latter is helpful to encourage the VAE in the first iterations to just put everything in the latent space
        # We create a mask below for which intervention targets are supposed to be predicted from which set of latents
        loss_mask = torch.cat([torch.eye(num_classes, dtype=torch.bool, device=target.device),       # Latent to causal variables
                               torch.zeros(1, num_classes, dtype=torch.bool, device=target.device),  # 'No-causal-var' slot
                               torch.ones(1, num_classes, dtype=torch.bool, device=target.device)    # 'All-var' slot
                              ], dim=0)
        loss_mask = loss_mask[None].expand(batch_size * time_steps, -1, -1).flatten(0, 1)

        # Model step => Cross entropy loss on targets for all sets of latents
        z_sample_model = z_sample.detach()
        target_assignment_det = target_assignment.detach()
        model_inps = torch.cat([z_sample_model[:,:-1,:], z_sample_model[:,1:,:] * target_assignment_det, target_assignment_det], dim=-1)
        model_inps = model_inps.flatten(0, 2)
        model_pred = self.classifiers(model_inps)
        loss_model = F.binary_cross_entropy_with_logits(model_pred, exp_target, reduction='none')
        loss_model = num_targets * time_steps * loss_model.mean()

        # Log target classification accuracies
        if logger is not None:
            with torch.no_grad():
                acc = ((model_pred > 0.0).float() == exp_target).float().unflatten(0, (batch_size * time_steps, num_targets)).mean(dim=0)
                for b in range(num_targets):
                    for c in range(num_classes):
                        logger.log(f'target_classifier_block{self._tag_to_str(b)}_class{self._tag_to_str(c)}', acc[b,c], on_step=False, on_epoch=True)

        # Latent step => Cross entropy loss on true targets for respective sets of latents, and cross entropy loss on marginal (conditional) accuracy otherwise.
        z_inps = torch.cat([z_sample[:,:-1,:], z_sample[:,1:,:] * target_assignment, target_assignment], dim=-1)
        z_inps = z_inps.flatten(0, 2)
        z_pred = self.exp_classifiers(z_inps)
        z_pred_unflatten = z_pred.unflatten(0, (batch_size * time_steps, num_targets))
        if hasattr(self, 'avg_cond_dist'):
            avg_dist_labels = self.avg_cond_dist[None,None,:,:,1,1] * target[:,:,None,:] + self.avg_cond_dist[None,None,:,:,0,1] * (1 - target[:,:,None,:])
            avg_dist_labels = avg_dist_labels.permute(0, 1, 3, 2).flatten(0, 1)
            avg_dist_labels = torch.cat([avg_dist_labels, self.avg_dist[None,None,:,1].expand(avg_dist_labels.shape[0], 2, -1)], dim=1)
            avg_dist_labels = avg_dist_labels.flatten(0, 1)
        else:
            avg_dist_labels = self.avg_dist[None,:,1]
        z_targets = torch.where(loss_mask, exp_target, avg_dist_labels)
        loss_z = F.binary_cross_entropy_with_logits(z_pred, z_targets, reduction='none')
        loss_mask = loss_mask.float()
        pos_weight = num_classes  # Beneficial to weight the cross entropy loss for true target higher, especially for many causal variables
        loss_z = loss_z * (pos_weight * loss_mask + (1 - loss_mask))
        loss_z = loss_z.mean()
        loss_z = num_targets * time_steps * loss_z

        return loss_model, loss_z