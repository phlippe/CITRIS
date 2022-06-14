"""
Helper functions used for calculations in multiple spots
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import os
from scipy.stats import norm


def get_act_fn(act_fn_name):
    """ Map activation function string to activation function """
    act_fn_name = act_fn_name.lower()
    if act_fn_name == 'elu':
        act_fn_func = nn.ELU
    elif act_fn_name == 'silu':
        act_fn_func = nn.SiLU
    elif act_fn_name == 'leakyrelu':
        act_fn_func = lambda: nn.LeakyReLU(negative_slope=0.05, inplace=True)
    elif act_fn_name == 'relu':
        act_fn_func = nn.ReLU
    else:
        assert False, f'Unknown activation function \"{act_fn_name}\"'
    return act_fn_func


def kl_divergence(mean1, log_std1, mean2=None, log_std2=None): 
    """ Returns the KL divergence between two Gaussian distributions """
    if mean2 is None:
        mean2 = torch.zeros_like(mean1)
    if log_std2 is None:
        log_std2 = torch.zeros_like(log_std1)

    var1, var2 = (2*log_std1).exp(), (2*log_std2).exp()
    KLD = (log_std2 - log_std1) + (var1 + (mean1 - mean2) ** 2) / (2 * var2) - 0.5
    return KLD


def general_kl_divergence(samples=None, log_prob_q=None, log_prob_p=None, log_q=None, log_p=None, sample_dim=-1):
    """ Returns the approximated KL divergence between two arbitrary distributions based on samples """
    if log_q is None:
        log_q = log_prob_q(samples)
    if log_p is None:
        log_p = log_prob_p(samples)
    KLD = - (log_p - log_q).mean(dim=sample_dim)
    return KLD


def gaussian_log_prob(mean, log_std, samples):
    """ Returns the log probability of a specified Gaussian for a tensor of samples """
    if len(samples.shape) == len(mean.shape)+1:
        mean = mean[...,None]
    if len(samples.shape) == len(log_std.shape)+1:
        log_std = log_std[...,None]
    return - log_std - 0.5 * np.log(2*np.pi) - 0.5 * ((samples - mean) / log_std.exp())**2


def gaussian_mixture_log_prob(means, log_stds, mixture_log_probs, samples):
    """ Returns the log probability for a multi-modal Gaussian """
    mixture_log_probs = F.log_softmax(mixture_log_probs, dim=-1)
    if samples.shape[-2] == means.shape[-3]:
        logsum_dim = -1
        samples = samples[...,None]
    else:
        logsum_dim = -2
        samples = samples[...,None,:]
        mixture_log_probs = mixture_log_probs[...,None]
    log_probs = gaussian_log_prob(means, log_stds, samples)
    log_probs = log_probs + mixture_log_probs
    return torch.logsumexp(log_probs, dim=logsum_dim)


def evaluate_adj_matrix(pred_adj_matrix, gt_adj_matrix):
    pred_adj_matrix = pred_adj_matrix.bool()
    gt_adj_matrix = gt_adj_matrix.bool()

    false_positives = torch.logical_and(pred_adj_matrix, ~gt_adj_matrix)
    false_negatives = torch.logical_and(~pred_adj_matrix, gt_adj_matrix)
    TP = torch.logical_and(pred_adj_matrix, gt_adj_matrix).float().sum().item()
    TN = torch.logical_and(~pred_adj_matrix, ~gt_adj_matrix).float().sum().item()
    FP = false_positives.float().sum().item()
    FN = false_negatives.float().sum().item()
    TN = TN - pred_adj_matrix.shape[-1]  # Remove diagonal as those are not being predicted
    recall = TP / max(TP + FN, 1e-5)
    precision = TP / max(TP + FP, 1e-5)
    # Structural Hamming Distance score
    rev = torch.logical_and(pred_adj_matrix, gt_adj_matrix.T)
    num_revs = rev.float().sum().item()
    SHD = (false_positives + false_negatives + rev + rev.T).float().sum().item() - num_revs

    metrics = {
        "SHD": int(SHD),
        "recall": recall,
        "precision": precision
    }
    return metrics


@torch.no_grad()
def add_ancestors_to_adj_matrix(adj_matrix, remove_diag=True, exclude_cycles=False):
    adj_matrix = adj_matrix.bool()
    orig_adj_matrix = adj_matrix
    eye_matrix = torch.eye(adj_matrix.shape[-1], device=adj_matrix.device, dtype=torch.bool).reshape((1,)*(len(adj_matrix.shape)-2) + (-1, adj_matrix.shape[-1]))
    changed = True
    while changed: 
        new_anc = torch.logical_and(adj_matrix[..., None], adj_matrix[...,None, :, :]).any(dim=-2)
        new_anc = torch.logical_or(adj_matrix, new_anc)
        changed = not (new_anc == adj_matrix).all().item()
        adj_matrix = new_anc
    if exclude_cycles:
        is_diagonal = torch.logical_and(adj_matrix, eye_matrix).any(dim=-2, keepdims=True)
        adj_matrix = torch.where(is_diagonal, orig_adj_matrix, adj_matrix)
        
    if remove_diag:
        adj_matrix = torch.logical_and(adj_matrix, ~eye_matrix)
    return adj_matrix.float()


def log_dict(d, name, current_epoch=None, log_dir=None, trainer=None):
    """ Saves a dictionary of numpy arrays to the logging directory """
    if log_dir is None:
        log_dir = trainer.logger.log_dir
    filename = os.path.join(log_dir, name + '.npz')

    new_epoch = trainer.current_epoch if current_epoch is None else current_epoch
    new_epoch = np.array([new_epoch])
    values = {key: np.array([d[key]]) for key in d}
    if os.path.isfile(filename):
        prev_data = np.load(filename)
        epochs = prev_data['epochs']
        epochs = np.concatenate([epochs, new_epoch], axis=0)
        for key in values:
            values[key] = np.concatenate([prev_data[key], values[key]], axis=0)
    else:
        epochs = new_epoch
    np.savez_compressed(filename, epochs=epochs, **values)


def log_matrix(matrix, trainer, name, current_epoch=None, log_dir=None):
    """ Saves a numpy array to the logging directory """
    if log_dir is None:
        log_dir = trainer.logger.log_dir
    filename = os.path.join(log_dir, name + '.npz')

    new_epoch = trainer.current_epoch if current_epoch is None else current_epoch
    new_epoch = np.array([new_epoch])
    new_val = matrix[None]
    if os.path.isfile(filename):
        prev_data = np.load(filename)
        epochs, values = prev_data['epochs'], prev_data['values']
        epochs = np.concatenate([epochs, new_epoch], axis=0)
        values = np.concatenate([values, new_val], axis=0)
    else:
        epochs = new_epoch
        values = new_val
    np.savez_compressed(filename, epochs=epochs, values=values)