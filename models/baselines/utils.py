import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import numpy as np
from collections import defaultdict

import sys
sys.path.append('../')
from models.shared import CosineWarmupScheduler, SineWarmupScheduler, get_act_fn, kl_divergence, log_dict, Encoder, Decoder
from models.shared import ImageLogCallback, CorrelationMetricsLogCallback, CausalEncoder, PositionLayer
from experiments.datasets import Causal3DDataset, PinballDataset


class BaselineCorrelationMetricsLogCallback(CorrelationMetricsLogCallback):
    """ 
    Adapting the correlation metrics callback to the baselines by first running 
    the correlation estimation for every single latent variable, and then grouping
    them according to the highest correlation with a causal variable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.test_dataset is None:
            self.test_dataset = self.val_dataset

    @torch.no_grad()
    def on_validation_epoch_start(self, trainer, pl_module, is_test=False):
        self.log_postfix = '_all_latents' + ('_test' if is_test else '')
        self.extra_postfix = '_test' if is_test else ''
        pl_module.target_assignment = None
        r2_matrix = self.test_model(trainer, pl_module)
        max_r2 = torch.from_numpy(r2_matrix).argmax(dim=-1)
        ta = F.one_hot(max_r2, num_classes=r2_matrix.shape[-1]).float()
        if isinstance(self.dataset, Causal3DDataset) and self.dataset.coarse_vars:
            ta = torch.cat([ta[:,:3].sum(dim=-1, keepdims=True), ta[:,3:5].sum(dim=-1, keepdims=True), ta[:,5:]], dim=-1)
        elif isinstance(self.dataset, PinballDataset):
            ta = torch.cat([ta[:,:4].sum(dim=-1, keepdims=True),
                            ta[:,4:9].sum(dim=-1, keepdims=True),
                            ta[:,9:]], dim=-1)
        pl_module.target_assignment = ta
        pl_module.last_target_assignment.data = ta

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module, is_test=False):
        self.log_postfix = '_grouped_latents' + ('_test' if is_test else '')
        self.test_model(trainer, pl_module)
        
        if not is_test:
            results = trainer._results
            if 'validation_step.val_loss' in results:
                val_comb_loss = results['validation_step.val_loss'].value / results['validation_step.val_loss'].cumulated_batch_size
                new_val_dict = {'triplet_loss': val_comb_loss}
                for key in ['on_validation_epoch_end.corr_callback_r2_matrix_diag_grouped_latents',
                            'on_validation_epoch_end.corr_callback_spearman_matrix_diag_grouped_latents',
                            'on_validation_epoch_end.corr_callback_r2_matrix_max_off_diag_grouped_latents',
                            'on_validation_epoch_end.corr_callback_spearman_matrix_max_off_diag_grouped_latents']:
                    if key in results:
                        val = results[key].value
                        new_val_dict[key.split('_',5)[-1]] = val
                        if 'matrix_diag' in key:
                            val = 1 - val
                        val_comb_loss += val
                pl_module.log(f'val_comb_loss{self.log_postfix}{self.extra_postfix}', val_comb_loss)
                new_val_dict = {key: (val.item() if isinstance(val, torch.Tensor) else val) for key, val in new_val_dict.items()}
                if self.cluster:
                    s = f'[Epoch {trainer.current_epoch}] ' + ', '.join([f'{key}: {new_val_dict[key]:5.3f}' for key in sorted(list(new_val_dict.keys()))])
                    print(s)

    @torch.no_grad()
    def on_test_epoch_start(self, trainer, pl_module):
        self.dataset = self.test_dataset
        self.on_validation_epoch_start(trainer, pl_module, is_test=True)

    @torch.no_grad()
    def on_test_epoch_end(self, trainer, pl_module):
        self.on_validation_epoch_end(trainer, pl_module, is_test=True)
        self.dataset = self.val_dataset