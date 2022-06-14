import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy

from models.shared import TargetClassifier, add_ancestors_to_adj_matrix


class InstantaneousTargetClassifier(TargetClassifier):
    """
    The target classifier for guiding towards better disentanglement of the causal variables in latent space.
    This is adapted for potentially instantaneous effects, since parents are needed to predict the interventions as well
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        target_counter = torch.zeros((2,)*self.num_blocks, dtype=torch.float32)
        torch_range = torch.arange(2**self.num_blocks, dtype=torch.float32)
        target_counter_mask = torch.stack([torch.div(torch_range, 2**i, rounding_mode='floor') % 2 for i in range(self.num_blocks-1, -1, -1)], dim=-1)
        self.register_buffer('target_counter', target_counter)
        self.register_buffer('target_counter_mask', target_counter_mask)
        self.register_buffer('graph_structures', target_counter_mask.clone())
        self.register_buffer('target_counter_prob', torch.zeros(2**self.num_blocks, 2**self.num_blocks, self.num_blocks))
        self.register_buffer('two_exp_range', torch.Tensor([2**i for i in range(self.num_blocks-1, -1, -1)]))

    @torch.no_grad()
    def _update_dist(self, target):
        super()._update_dist(target)
        if self.dist_steps < 1e6 and self.use_conditional_targets:
            target = target.flatten(0, -2)
            unique_targets, target_counts = torch.unique(target, dim=0, return_counts=True)
            self.target_counter[unique_targets.long().unbind(dim=-1)] += target_counts

            target_equal = (self.target_counter_mask[None,:] == self.target_counter_mask[:,None])
            mask = torch.logical_or(target_equal[None,:,:], self.graph_structures[:,None,None] == 0)
            mask = mask.all(dim=-1)
            mask = mask.reshape((mask.shape[0],) + (2,)*self.num_blocks + mask.shape[2:])
            masked_counter = mask * self.target_counter[None,...,None]
            all_probs = []
            for i in range(self.num_blocks):
                counter_sum = masked_counter.sum(dim=[j+1 for j in range(self.num_blocks) if i != j])
                counter_prob = counter_sum[:,1] / counter_sum.sum(dim=1).clamp_(min=1e-5)
                all_probs.append(counter_prob)
            self.target_counter_prob = torch.stack(all_probs, dim=-1)

    def forward(self, z_sample, target, transition_prior, logger=None, add_anc_prob=0.0):
        """
        Calculates the loss for the target classifier (predict all intervention targets from all sets of latents)
        and the latents + psi (predict only its respective intervention target from a set of latents).

        Parameters
        ----------
        z_sample : torch.FloatTensor, shape [batch_size, time_steps, num_latents]
                   The sets of latents for which the loss should be calculated. If time steps is 2, we 
                   use z^t=z_sample[:,0], z^t+1=z_sample[:,1]. If time steps is larger than 2, we apply
                   it for every possible pair over time.
        target : torch.FloatTensor, shape [batch_size, num_blocks]
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
        num_latents = z_sample.shape[-1]
        # Sample latent-to-causal variable assignments
        target_assignment = F.gumbel_softmax(transition_prior.target_params[None].expand(batch_size, time_steps, -1, -1), 
                                             tau=self.gumbel_temperature, hard=True)
        if target_assignment.shape[-1] == num_classes:  # No variable slot
            target_assignment = torch.cat([target_assignment, target_assignment.new_zeros(target_assignment.shape[:-1] + (1,))], dim=-1)
        target_assignment = torch.cat([target_assignment, target_assignment.new_ones(target_assignment.shape[:-1] + (1,))], dim=-1)  # All latents slot
        num_targets = target_assignment.shape[-1]
        target_assignment = target_assignment.transpose(-2, -1)  # shape [batch, time_steps, num_targets, latent_vars]
        z_sample = z_sample[...,None,:].expand(-1, -1, num_targets, -1)
        exp_target = target[...,None,:].expand(-1, -1, num_targets, -1).flatten(0, 2).float()

        # Sample adjacency matrices
        graph_probs = transition_prior.get_adj_matrix(hard=False, for_training=True)
        graph_samples = torch.bernoulli(graph_probs[None, None].expand(batch_size, time_steps, -1, -1))
        if add_anc_prob > 0.0:  # Once the graph is fixed, add ancestors to the mix here
            graph_samples_anc = add_ancestors_to_adj_matrix(graph_samples, remove_diag=True, exclude_cycles=True)
            graph_samples = torch.where(torch.rand(*graph_samples.shape[:2], 1, 1, device=graph_samples.device) < add_anc_prob, graph_samples_anc, graph_samples)
        # Add self-connections since we want to identify interventions from itself as well
        graph_samples_eye = graph_samples + torch.eye(graph_samples.shape[-1], device=graph_samples.device)[None, None]
        latent_to_causal = (target_assignment[:, :, :graph_probs.shape[0], :, None] * graph_samples_eye[:, :, :, None, :]).sum(dim=-3)
        latent_mask = latent_to_causal.transpose(-2, -1)  # shape: [batch_size, time_steps, causal_vars, latent_vars]
        latent_mask = torch.cat([latent_mask] +
                                ([latent_mask.new_zeros(batch_size, time_steps, 1, num_latents)] if (latent_mask.shape[2] == num_classes) else []) + 
                                 [latent_mask.new_ones(batch_size, time_steps, 1, num_latents)], 
                                 dim=-2)  # shape [batch, time_steps, num_targets, latent_vars]
        
        # Model step => Cross entropy loss on targets for all sets of latents
        z_sample_model = z_sample.detach()
        latent_mask_det = latent_mask.detach()
        model_inps = torch.cat([z_sample_model[:,:-1], z_sample_model[:,1:] * latent_mask_det, latent_mask_det], dim=-1)
        model_inps = model_inps.flatten(0, 2)
        model_pred = self.classifiers(model_inps)
        loss_model = F.binary_cross_entropy_with_logits(model_pred, exp_target, reduction='none')
        loss_model = num_targets * loss_model.mean()

        # Log target classification accuracies
        if logger is not None:
            with torch.no_grad():
                acc = ((model_pred > 0.0).float() == exp_target).float().unflatten(0, (batch_size * time_steps, num_targets)).mean(dim=0)
                for b in range(num_targets):
                    for c in range(num_classes):
                        logger.log(f'target_classifier_block{self._tag_to_str(b)}_class{self._tag_to_str(c)}', acc[b,c], on_step=False, on_epoch=True)
        
        # We consider 2 + [number of causal variables] sets of latents: 
        # (1) one per causal variable plus its children, (2) the noise/'no-causal-var' slot psi(0), (3) all latents
        # The latter is helpful to encourage the VAE in the first iterations to just put everything in the latent space
        # We create a mask below for which intervention targets are supposed to be predicted from which set of latents
        loss_mask = torch.eye(num_classes, dtype=torch.float32, device=target.device)
        loss_mask = loss_mask[None, None].expand(batch_size, time_steps, -1, -1)
        loss_mask = loss_mask - graph_samples.transpose(-2, -1)
        loss_mask = torch.cat([loss_mask,
                               torch.zeros(batch_size, time_steps, 1, num_classes, dtype=torch.float32, device=target.device),  # 'No-causal-var' slot
                               torch.ones(batch_size, time_steps, 1, num_classes, dtype=torch.float32, device=target.device)    # 'All-var' slot
                              ], dim=2)
        loss_mask = loss_mask.flatten(0, 2)

        # Latent step => Cross entropy loss on true targets for respective sets of latents, and cross entropy loss on marginal (conditional) accuracy otherwise.
        z_inps = torch.cat([z_sample[:,:-1], z_sample[:,1:] * latent_mask, latent_mask], dim=-1)
        z_inps = z_inps.flatten(0, 2)
        z_pred = self.exp_classifiers(z_inps)
        z_pred_unflatten = z_pred.unflatten(0, (batch_size * time_steps, num_targets))
        if hasattr(self, 'avg_cond_dist'):
            with torch.no_grad():
                # Add labels for all variables
                pred_logits = z_pred_unflatten.detach()[:,:-2]
                pred_probs = F.logsigmoid(pred_logits)
                pred_neg_probs = F.logsigmoid(-pred_logits)
                target_flat = target.flatten(0, 1)
                graphs_flat = graph_samples_eye.flatten(0, 1)
                graph_idxs = (graphs_flat * self.two_exp_range[None,:,None]).sum(dim=1).long()
                avg_dist_labels = self.target_counter_prob[graph_idxs]

                target_weights = torch.where(self.target_counter_mask[None,None] == 0, pred_neg_probs[:,:,None], pred_probs[:,:,None]).sum(dim=-1)
                target_weights = torch.softmax(target_weights, dim=-1)
                avg_dist_labels = (avg_dist_labels * target_weights[...,None]).sum(dim=-2)
                
                # Add labels for no and all variables
                avg_dist_labels = torch.cat([avg_dist_labels, 
                                             self.avg_dist[None,None,:,1].expand(avg_dist_labels.shape[0], 2, -1)], dim=1)
                avg_dist_labels = avg_dist_labels.flatten(0, 1)
        else:
            avg_dist_labels = self.avg_dist[None,:,1]
        z_targets = torch.where(loss_mask == 1, exp_target, avg_dist_labels)
        loss_z = F.binary_cross_entropy_with_logits(z_pred, z_targets, reduction='none')
        pos_weight = num_classes  # Beneficial to weight the cross entropy loss for true target higher, especially for many causal variables
        loss_z = loss_z * (pos_weight * (loss_mask == 1).float() + 1 * (loss_mask == 0).float())
        loss_z = loss_z.mean()
        loss_z = num_targets * loss_z

        return loss_model, loss_z