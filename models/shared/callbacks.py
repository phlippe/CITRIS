import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import OrderedDict, defaultdict
from tqdm.auto import tqdm
from scipy.stats import spearmanr

import sys
sys.path.append('../../')
from models.shared.visualization import visualize_reconstruction, plot_target_assignment, plot_target_classification, visualize_triplet_reconstruction, visualize_graph, plot_latents_mutual_information
from models.shared.utils import log_matrix, log_dict, evaluate_adj_matrix
from models.shared.causal_encoder import CausalEncoder
from models.shared.enco import ENCOGraphLearning


class ImageLogCallback(pl.Callback):
    """ Callback for creating visualizations for logging """

    def __init__(self, exmp_inputs, dataset, every_n_epochs=10, cluster=False, prefix=''):
        super().__init__()
        self.imgs = exmp_inputs[0]
        if len(exmp_inputs) > 2 and len(exmp_inputs[1].shape) == len(self.imgs.shape):
            self.labels = exmp_inputs[1]
            self.extra_inputs = exmp_inputs[2:]
        else:
            self.labels = self.imgs
            self.extra_inputs = exmp_inputs[1:]
        self.dataset = dataset
        self.every_n_epochs = every_n_epochs
        self.prefix = prefix
        self.cluster = cluster

    def on_train_epoch_end(self, trainer, pl_module):
        def log_fig(tag, fig):
            if fig is None:
                return
            trainer.logger.experiment.add_figure(f'{self.prefix}{tag}', fig, global_step=trainer.global_step)
            plt.close(fig)

        if hasattr(trainer.model, 'intv_classifier'):
            if (trainer.current_epoch+1) % (2 if not self.cluster else self.every_n_epochs) == 0:
                log_fig(f'target_classifier', plot_target_classification(trainer._results))

        if hasattr(trainer.model, 'prior_t1'):
            if hasattr(trainer.model, 'prior_t1') and (trainer.current_epoch+1) % self.every_n_epochs == 0:
                log_fig('target_assignment', plot_target_assignment(trainer.model.prior_t1, dataset=self.dataset))

        if self.imgs is not None and (trainer.current_epoch+1) % self.every_n_epochs == 0:
            trainer.model.eval()
            images = self.imgs.to(trainer.model.device)
            labels = self.labels.to(trainer.model.device)
            if len(images.shape) == 5:
                full_imgs, full_labels = images, labels
                images = images[:,0]
                labels = labels[:,0]
            else:
                full_imgs, full_labels = None, None

            for i in range(min(4, images.shape[0])):
                log_fig(f'reconstruction_{i}', visualize_reconstruction(trainer.model, images[i], labels[i], self.dataset))
            
            if hasattr(trainer.model, 'prior_t1'):
                if full_imgs is not None:
                    for i in range(min(4, full_imgs.shape[0])):
                        log_fig(f'triplet_visualization_{i}', visualize_triplet_reconstruction(trainer.model, full_imgs[i], full_labels[i], [e[i] for e in self.extra_inputs], dataset=self.dataset))
            trainer.model.train()


class CorrelationMetricsLogCallback(pl.Callback):
    """ Callback for extracting correlation metrics (R^2 and Spearman) """

    def __init__(self, dataset, every_n_epochs=10, num_train_epochs=100, cluster=False, test_dataset=None):
        super().__init__()
        assert dataset is not None, "Dataset for correlation metrics cannot be None."
        self.dataset = dataset
        self.val_dataset = dataset
        self.test_dataset = test_dataset
        self.every_n_epochs = every_n_epochs
        self.num_train_epochs = num_train_epochs
        self.cluster = cluster
        self.log_postfix = ''
        self.extra_postfix = ''

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        if isinstance(self.dataset, dict):
            dataset_dict = self.dataset
            data_len_sum = sum([len(dataset_dict[key]) for key in dataset_dict])
            for key in dataset_dict:
                self.dataset = dataset_dict[key]
                self.log_postfix = f'{self.extra_postfix}_{key}'
                self.test_model(trainer, pl_module)
            self.log_postfix = ''
            self.dataset = dataset_dict
        else:
            self.test_model(trainer, pl_module)

        results = trainer._results
        if 'validation_step.val_loss' in results:
            val_comb_loss = results['validation_step.val_loss'].value / results['validation_step.val_loss'].cumulated_batch_size
            new_val_dict = {'triplet_loss': val_comb_loss}
            for key in ['on_validation_epoch_end.corr_callback_r2_matrix_diag',
                        'on_validation_epoch_end.corr_callback_spearman_matrix_diag',
                        'on_validation_epoch_end.corr_callback_r2_matrix_max_off_diag',
                        'on_validation_epoch_end.corr_callback_spearman_matrix_max_off_diag']:
                if key in results:
                    val = results[key].value
                    new_val_dict[key.split('_',5)[-1]] = val
                    if key.endswith('matrix_diag'):
                        val = 1 - val
                    val_comb_loss += val
            pl_module.log(f'val_comb_loss{self.log_postfix}{self.extra_postfix}', val_comb_loss)
            new_val_dict = {key: (val.item() if isinstance(val, torch.Tensor) else val) for key, val in new_val_dict.items()}
            if self.cluster:
                s = f'[Epoch {trainer.current_epoch}] ' + ', '.join([f'{key}: {new_val_dict[key]:5.3f}' for key in sorted(list(new_val_dict.keys()))])
                print(s)

    @torch.no_grad()
    def on_test_epoch_end(self, trainer, pl_module):
        if self.test_dataset is None:
            print('Skipping correlation metrics testing due to missing dataset...')
        else:
            val_dataset = self.dataset
            self.dataset = self.test_dataset
            self.log_postfix = '_test'
            self.extra_postfix = '_test'
            self.on_validation_epoch_end(trainer, pl_module)
            self.dataset = val_dataset
            self.log_postfix = ''
            self.extra_postfix = ''

    @torch.no_grad()
    def test_model(self, trainer, pl_module):
        # Encode whole dataset with pl_module
        is_training = pl_module.training
        pl_module = pl_module.eval()
        loader = data.DataLoader(self.dataset, batch_size=256, drop_last=False, shuffle=False)
        all_encs, all_latents = [], []
        for batch in loader:
            inps, *_, latents = batch
            encs = pl_module.encode(inps.to(pl_module.device)).cpu()
            all_encs.append(encs)
            all_latents.append(latents)
        all_encs = torch.cat(all_encs, dim=0)
        all_latents = torch.cat(all_latents, dim=0)
        # Normalize latents for stable gradient signals
        all_encs = (all_encs - all_encs.mean(dim=0, keepdim=True)) / all_encs.std(dim=0, keepdim=True).clamp(min=1e-2)
        # Create new tensor dataset for training (50%) and testing (50%)
        full_dataset = data.TensorDataset(all_encs, all_latents)
        train_size = int(0.5 * all_encs.shape[0])
        test_size = all_encs.shape[0] - train_size
        train_dataset, test_dataset = data.random_split(full_dataset, 
                                                        lengths=[train_size, test_size], 
                                                        generator=torch.Generator().manual_seed(42))
        # Train network to predict causal factors from latent variables
        if hasattr(pl_module, 'prior_t1'):
            target_assignment = pl_module.prior_t1.get_target_assignment(hard=True)
        elif hasattr(pl_module, 'target_assignment') and pl_module.target_assignment is not None:
            target_assignment = pl_module.target_assignment.clone()
        else:
            target_assignment = torch.eye(all_encs.shape[-1])
        encoder = self.train_network(pl_module, train_dataset, target_assignment)
        encoder.eval()
        # Record predictions of model on test and calculate distances
        test_inps, test_labels = all_encs[test_dataset.indices], all_latents[test_dataset.indices]
        test_exp_inps, test_exp_labels = self._prepare_input(test_inps, target_assignment.cpu(), test_labels, flatten_inp=False)
        pred_dict = encoder.forward(test_exp_inps.to(pl_module.device))
        for key in pred_dict:
            pred_dict[key] = pred_dict[key].cpu()
        _, dists, norm_dists = encoder.calculate_loss_distance(pred_dict, test_exp_labels)
        # Calculate statistics (R^2, pearson, etc.)
        avg_norm_dists, r2_matrix = self.log_R2_statistic(trainer, encoder, pred_dict, test_labels, norm_dists, pl_module=pl_module)
        self.log_Spearman_statistics(trainer, encoder, pred_dict, test_labels, pl_module=pl_module)
        if is_training:
            pl_module = pl_module.train()
        return r2_matrix

    def train_network(self, pl_module, train_dataset, target_assignment):
        device = pl_module.device
        if hasattr(pl_module, 'causal_encoder'):
            causal_var_info = pl_module.causal_encoder.hparams.causal_var_info
        else:
            causal_var_info = pl_module.hparams.causal_var_info
        # We use one, sufficiently large network that predicts for any input all causal variables
        # To iterate over the different sets, we use a mask which is an extra input to the model
        # This is more efficient than using N networks and showed same results with large hidden size
        encoder = CausalEncoder(c_hid=128,
                                lr=4e-3,
                                causal_var_info=causal_var_info,
                                single_linear=True,
                                c_in=pl_module.hparams.num_latents*2,
                                warmup=0)
        optimizer, _ = encoder.configure_optimizers()
        if isinstance(optimizer, (list, tuple)):
            optimizer = optimizer[0]

        train_loader = data.DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=512)
        target_assignment = target_assignment.to(device)
        encoder.to(device)
        encoder.train()
        with torch.enable_grad():
            range_iter = range(self.num_train_epochs)
            if not self.cluster:
                range_iter = tqdm(range_iter, leave=False, desc=f'Training correlation encoder {self.log_postfix}')
            for epoch_idx in range_iter:
                avg_loss = 0.0
                for inps, latents in train_loader:
                    inps = inps.to(device)
                    latents = latents.to(device)
                    inps, latents = self._prepare_input(inps, target_assignment, latents)
                    loss = encoder._get_loss([inps, latents], mode=None)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    avg_loss += loss.item()
        return encoder

    def _prepare_input(self, inps, target_assignment, latents, flatten_inp=True):
        ta = target_assignment.detach()[None,:,:].expand(inps.shape[0], -1, -1)
        inps = torch.cat([inps[:,:,None] * ta, ta], dim=-2).permute(0, 2, 1)
        latents = latents[:,None].expand(-1, inps.shape[1], -1)
        if flatten_inp:
            inps = inps.flatten(0, 1)
            latents = latents.flatten(0, 1)
        return inps, latents

    def log_R2_statistic(self, trainer, encoder, pred_dict, test_labels, norm_dists, pl_module=None):
        avg_pred_dict = OrderedDict()
        for i, var_key in enumerate(encoder.hparams.causal_var_info):
            var_info = encoder.hparams.causal_var_info[var_key]
            gt_vals = test_labels[...,i]
            if var_info.startswith('continuous'):
                avg_pred_dict[var_key] = gt_vals.mean(dim=0, keepdim=True).expand(gt_vals.shape[0],)
            elif var_info.startswith('angle'):
                avg_angle = torch.atan2(torch.sin(gt_vals).mean(dim=0, keepdim=True), 
                                        torch.cos(gt_vals).mean(dim=0, keepdim=True)).expand(gt_vals.shape[0],)
                avg_angle = torch.where(avg_angle < 0.0, avg_angle + 2*np.pi, avg_angle)
                avg_pred_dict[var_key] = torch.stack([torch.sin(avg_angle), torch.cos(avg_angle)], dim=-1)
            elif var_info.startswith('categ'):
                gt_vals = gt_vals.long()
                mode = torch.mode(gt_vals, dim=0, keepdim=True).values
                avg_pred_dict[var_key] = F.one_hot(mode, int(var_info.split('_')[-1])).float().expand(gt_vals.shape[0], -1)
            else:
                assert False, f'Do not know how to handle key \"{var_key}\" in R2 statistics.'
        _, _, avg_norm_dists = encoder.calculate_loss_distance(avg_pred_dict, test_labels, keep_sign=True)

        r2_matrix = []
        for var_key in encoder.hparams.causal_var_info:
            ss_res = (norm_dists[var_key] ** 2).mean(dim=0)
            ss_tot = (avg_norm_dists[var_key] ** 2).mean(dim=0, keepdim=True)
            r2 = 1 - ss_res / ss_tot
            r2_matrix.append(r2)
        r2_matrix = torch.stack(r2_matrix, dim=-1).cpu().numpy()
        log_matrix(r2_matrix, trainer, 'r2_matrix' + self.log_postfix)
        self._log_heatmap(trainer=trainer, 
                          values=r2_matrix, 
                          tag='r2_matrix',
                          title='R^2 Matrix',
                          xticks=[key for key in encoder.hparams.causal_var_info],
                          pl_module=pl_module)
        return avg_norm_dists, r2_matrix

    def log_pearson_statistic(self, trainer, encoder, pred_dict, test_labels, norm_dists, avg_gt_norm_dists, pl_module=None):
        avg_pred_dict = OrderedDict()
        for i, var_key in enumerate(encoder.hparams.causal_var_info):
            var_info = encoder.hparams.causal_var_info[var_key]
            pred_vals = pred_dict[var_key]
            if var_info.startswith('continuous'):
                pred_vals = pred_vals.squeeze(dim=-1)
                avg_pred_dict[var_key] = pred_vals.mean(dim=0, keepdim=True).expand(pred_vals.shape[0], -1)
            elif var_info.startswith('angle'):
                angles = torch.atan(pred_vals[...,0] / pred_vals[...,1])
                avg_angle = torch.atan2(torch.sin(angles).mean(dim=0, keepdim=True), 
                                        torch.cos(angles).mean(dim=0, keepdim=True)).expand(pred_vals.shape[0], -1)
                avg_angle = torch.where(avg_angle < 0.0, avg_angle + 2*np.pi, avg_angle)
                avg_pred_dict[var_key] = avg_angle
            elif var_info.startswith('categ'):
                pred_vals = pred_vals.argmax(dim=-1)
                mode = torch.mode(pred_vals, dim=0, keepdim=True).values
                avg_pred_dict[var_key] = mode.expand(pred_vals.shape[0], -1)
            else:
                assert False, f'Do not know how to handle key \"{var_key}\" in Pearson statistics.'
        _, _, avg_pred_norm_dists = encoder.calculate_loss_distance(pred_dict, gt_vec=torch.stack([avg_pred_dict[key] for key in avg_pred_dict], dim=-1), keep_sign=True)

        pearson_matrix = []
        for var_key in encoder.hparams.causal_var_info:
            var_info = encoder.hparams.causal_var_info[var_key]
            pred_dist, gt_dist = avg_pred_norm_dists[var_key], avg_gt_norm_dists[var_key]
            nomin = (pred_dist * gt_dist[:,None]).sum(dim=0)
            denom = torch.sqrt((pred_dist**2).sum(dim=0) * (gt_dist[:,None]**2).sum(dim=0))
            p = nomin / denom.clamp(min=1e-5)
            pearson_matrix.append(p)
        pearson_matrix = torch.stack(pearson_matrix, dim=-1).cpu().numpy()
        log_matrix(pearson_matrix, trainer, 'pearson_matrix' + self.log_postfix)
        self._log_heatmap(trainer=trainer, 
                          values=pearson_matrix, 
                          tag='pearson_matrix',
                          title='Pearson Matrix',
                          xticks=[key for key in encoder.hparams.causal_var_info],
                          pl_module=pl_module)

    def log_Spearman_statistics(self, trainer, encoder, pred_dict, test_labels, pl_module=None):
        spearman_matrix = []
        for i, var_key in enumerate(encoder.hparams.causal_var_info):
            var_info = encoder.hparams.causal_var_info[var_key]
            gt_vals = test_labels[...,i]
            pred_val = pred_dict[var_key]
            if var_info.startswith('continuous'):
                spearman_preds = pred_val.squeeze(dim=-1)  # Nothing needs to be adjusted
            elif var_info.startswith('angle'):
                spearman_preds = F.normalize(pred_val, p=2.0, dim=-1)
                gt_vals = torch.stack([torch.sin(gt_vals), torch.cos(gt_vals)], dim=-1)
            elif var_info.startswith('categ'):
                spearman_preds = pred_val.argmax(dim=-1).float()
            else:
                assert False, f'Do not know how to handle key \"{var_key}\" in Spearman statistics.'

            gt_vals = gt_vals.cpu().numpy()
            spearman_preds = spearman_preds.cpu().numpy()
            results = torch.zeros(spearman_preds.shape[1],)
            for j in range(spearman_preds.shape[1]):
                if len(spearman_preds.shape) == 2:
                    if np.unique(spearman_preds[:,j]).shape[0] == 1:
                        results[j] = 0.0
                    else:
                        results[j] = spearmanr(spearman_preds[:,j], gt_vals).correlation
                elif len(spearman_preds.shape) == 3:
                    num_dims = spearman_preds.shape[-1]
                    for k in range(num_dims):
                        if np.unique(spearman_preds[:,j,k]).shape[0] == 1:
                            results[j] = 0.0
                        else:
                            results[j] += spearmanr(spearman_preds[:,j,k], gt_vals[...,k]).correlation
                    results[j] /= num_dims
                
            spearman_matrix.append(results)
        
        spearman_matrix = torch.stack(spearman_matrix, dim=-1).cpu().numpy()
        log_matrix(spearman_matrix, trainer, 'spearman_matrix' + self.log_postfix)
        self._log_heatmap(trainer=trainer, 
                          values=spearman_matrix, 
                          tag='spearman_matrix',
                          title='Spearman\'s Rank Correlation Matrix',
                          xticks=[key for key in encoder.hparams.causal_var_info],
                          pl_module=pl_module)

    def _log_heatmap(self, trainer, values, tag, title=None, xticks=None, yticks=None, xlabel=None, ylabel=None, pl_module=None):
        if ylabel is None:
            ylabel = 'Target dimension'
        if xlabel is None:
            xlabel = 'True causal variable'
        if yticks is None:
            yticks = self.dataset.target_names()+['No variable']
            if values.shape[0] > len(yticks):
                yticks = [f'Dim {i+1}' for i in range(values.shape[0])]
            if len(yticks) > values.shape[0]:
                yticks = yticks[:values.shape[0]]
        if xticks is None:
            xticks = self.dataset.target_names()
        fig = plt.figure(figsize=(min(6, values.shape[1]/1.25), min(6, values.shape[0]/1.25)))
        sns.heatmap(values, annot=True,
                    yticklabels=yticks,
                    xticklabels=xticks,
                    fmt='3.2f')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title is not None:
            plt.title(title)
        plt.tight_layout()

        trainer.logger.experiment.add_figure(tag + self.log_postfix, fig, global_step=trainer.global_step)
        plt.close(fig)

        if values.shape[0] == values.shape[1] + 1:  # Remove 'lambda_sparse' variables
            values = values[:-1]

        if values.shape[0] == values.shape[1]:
            avg_diag = np.diag(values).mean()
            max_off_diag = (values - np.eye(values.shape[0]) * 10).max(axis=-1).mean()
            if pl_module is None:
                trainer.logger.experiment.add_scalar(f'corr_callback_{tag}_diag{self.log_postfix}', avg_diag, global_step=trainer.global_step)
                trainer.logger.experiment.add_scalar(f'corr_callback_{tag}_max_off_diag{self.log_postfix}', max_off_diag, global_step=trainer.global_step)
            else:
                pl_module.log(f'corr_callback_{tag}_diag{self.log_postfix}', avg_diag)
                pl_module.log(f'corr_callback_{tag}_max_off_diag{self.log_postfix}', max_off_diag)


class GraphLogCallback(pl.Callback):
    """ Callback for creating visualizations for logging """

    def __init__(self, every_n_epochs=1, dataset=None, cluster=False):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        if dataset is not None and hasattr(dataset, 'get_adj_matrix'):
            self.gt_adj_matrix = dataset.get_adj_matrix()
        else:
            self.gt_adj_matrix = None
        self.last_adj_matrix = None
        self.cluster = cluster
        self.log_string = None

    def on_train_epoch_end(self, trainer, pl_module):
        if hasattr(trainer.model, 'prior_t1') and hasattr(trainer.model.prior_t1, 'get_adj_matrix'):
            adj_matrix = trainer.model.prior_t1.get_adj_matrix(hard=True).cpu().detach()
            log_matrix(adj_matrix.numpy(), trainer, 'instantaneous_adjacency_matrix')
            if (trainer.current_epoch+1) % self.every_n_epochs == 0:
                if self.last_adj_matrix is None or (self.last_adj_matrix != adj_matrix).any():
                    # Don't visualize the same graph several times, reduces tensorboard size
                    self.last_adj_matrix = adj_matrix
                    if hasattr(trainer.model.hparams, 'var_names'):
                        var_names = trainer.model.hparams.var_names
                    else:
                        var_names = []
                    while len(var_names) < adj_matrix.shape[0]-1:
                        var_names.append(f'C{len(var_names)}')
                    if len(var_names) == adj_matrix.shape[0]-1:
                        var_names.append('Noise')

                    fig = visualize_graph(nodes=var_names, adj_matrix=adj_matrix)
                    trainer.logger.experiment.add_figure('instantaneous_graph', fig, global_step=trainer.global_step)
                    plt.close(fig)

            if self.gt_adj_matrix is not None:
                metrics = evaluate_adj_matrix(adj_matrix, self.gt_adj_matrix)
                log_dict(metrics, 'instantaneous_adjacency_matrix_metrics', trainer=trainer)
                for key in metrics:
                    trainer.logger.experiment.add_scalar(f'adj_matrix_{key}', metrics[key], global_step=trainer.global_step)
                self.log_string = f'[Epoch {trainer.current_epoch+1}] ' + ', '.join([f'{key}: {metrics[key]:4.2f}' for key in sorted(list(metrics.keys()))])

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.cluster and self.log_string is not None:
            print(self.log_string)
            self.log_string = None
    
    def on_test_epoch_start(self, trainer, pl_module):
        if hasattr(trainer.model, 'prior_t1') and hasattr(trainer.model.prior_t1, 'get_adj_matrix'):
            adj_matrix = trainer.model.prior_t1.get_adj_matrix(hard=True).cpu().detach()
            if self.gt_adj_matrix is not None:
                metrics = evaluate_adj_matrix(adj_matrix, self.gt_adj_matrix)
                for key in metrics:
                    pl_module.log(f'test_adj_matrix_{key}', torch.FloatTensor([metrics[key]]))


class SparsifyingGraphCallback(pl.Callback):
    """ Callback for creating visualizations for logging """

    def __init__(self, dataset, lambda_sparse=[0.02], cluster=False, prefix=''):
        super().__init__()
        self.lambda_sparse = lambda_sparse
        self.cluster = cluster
        self.dataset = dataset
        self.prefix = prefix
        self.gt_adj_matrix = None
        self.gt_temporal_adj_matrix = None
        if dataset is not None and hasattr(dataset, 'get_adj_matrix'):
            self.gt_adj_matrix = dataset.get_adj_matrix()
        if dataset is not None and hasattr(dataset, 'get_temporal_adj_matrix'):
            self.gt_temporal_adj_matrix = dataset.get_temporal_adj_matrix()

    def set_test_prefix(self, prefix):
        self.prefix = '_' + prefix
        
    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def on_test_epoch_end(self, trainer, pl_module):
        pl_module = pl_module.eval()
        self.dataset.seq_len = 2
        logs = {}
        log_filename = os.path.join(trainer.logger.log_dir, f'enco_adj_matrices{self.prefix}.npz')

        lambda_sparse = self.lambda_sparse
        if not self.cluster:
            lambda_sparse = tqdm(lambda_sparse, desc='ENCO sparsity settings', leave=False)
        for l_sparse in lambda_sparse:
            enco = ENCOGraphLearning(model=pl_module,
                                     verbose=not self.cluster,
                                     lambda_sparse=l_sparse)
            temporal_adj_matrix, instantaneous_adj_matrix = enco.learn_graph(self.dataset)
            if self.gt_adj_matrix is not None:
                metrics = evaluate_adj_matrix(instantaneous_adj_matrix, self.gt_adj_matrix)
                for key in metrics:
                    pl_module.log(f'test{self.prefix}_instant_graph_{key}_lambda_sparse_{l_sparse}', torch.Tensor([metrics[key]]))
                    logs[f'{l_sparse}_instantaneous_{key}'] = np.array([metrics[key]])
            if self.gt_temporal_adj_matrix is not None:
                metrics_temporal = evaluate_adj_matrix(temporal_adj_matrix, self.gt_temporal_adj_matrix)
                for key in metrics_temporal:
                    pl_module.log(f'test{self.prefix}_temporal_graph_{key}_lambda_sparse_{l_sparse}', torch.Tensor([metrics_temporal[key]]))
                    logs[f'{l_sparse}_temporal_{key}'] = np.array([metrics_temporal[key]])
            logs[f'{l_sparse}_temporal_adj_matrix'] = temporal_adj_matrix
            logs[f'{l_sparse}_instantaneous_adj_matrix'] = instantaneous_adj_matrix
            np.savez_compressed(log_filename, **logs)
            visualize_graph(nodes=None, adj_matrix=temporal_adj_matrix)
            plt.savefig(os.path.join(trainer.logger.log_dir, f'temporal_graph_lsparse_{str(l_sparse).replace(".", "_")}{self.prefix}.pdf'))
            plt.close()
            visualize_graph(nodes=None, adj_matrix=instantaneous_adj_matrix)
            plt.savefig(os.path.join(trainer.logger.log_dir, f'instantaneous_graph_lsparse_{str(l_sparse).replace(".", "_")}{self.prefix}.pdf'))
            plt.close()