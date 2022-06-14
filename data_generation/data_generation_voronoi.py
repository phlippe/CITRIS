"""
Create the Voronoi benchmark including rendering.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import os
import math
from imageio import imread
from tqdm.auto import tqdm
import json
from multiprocessing import Pool
import time
from glob import glob
from argparse import ArgumentParser
from copy import deepcopy
import torch
import torch.nn as nn
from scipy.spatial import Voronoi

import sys
sys.path.append('../')

from models.shared import visualize_graph
from models.shared import AutoregNormalizingFlow, ActNormFlow, OrthogonalFlow


def soft_limit(v, v_min, v_max):
    """
    Limit a value 'v' between v_min and v_max.
    """
    v = torch.tanh(v)
    v = (v + 1.0) / 2.0
    v = v * (v_max - v_min) + v_min
    return v


def limit_inverse(v, v_min, v_max):
    """
    Inverse of the function soft_limit
    """
    v = (v - v_min) / (v_max - v_min)
    v = (v * 2.0) - 1.0
    v = torch.atanh(v.clamp_(min=-0.99, max=0.99))
    return v


def next_step(prev_time_step, settings):
    """
    Combines the observational and interventional dynamics to sample a new step with respecting the intervention sample policy.
    """
    intv_targets = {}
    keys = [f'c{i}' for i in range(settings['num_causal_vars'])]
    num_vars = len(keys)
    if settings['single_target_interventions']:
        t = np.random.randint(num_vars)
        no_int_prob = 2. / (num_vars + 2)
        if np.random.uniform() < no_int_prob:
            t = -1
        intv_targets = {key: int(t == i) for i, key in enumerate(keys)}
    elif settings['grouped_target_interventions']:
        num_groups = len(settings['intervention_groups'])
        if settings['intv_prob'] > 0.0:
            no_int_prob = 1 - settings['intv_prob']
            t = np.random.randint(num_groups - 1) if np.random.uniform() > no_int_prob else -1
        else:
            t = np.random.randint(num_groups)
        intv_targets = settings['intervention_groups'][t]
    else:
        intv_targets = {key: int(np.random.rand() < settings['intv_prob']) for key in keys}

    time_step = {}
    vals = sample_network_data(prev_time_step, intv_targets, settings).squeeze(dim=0)
    for i, k in enumerate(keys):
        v = soft_limit(vals[i], settings[f'{k}_min'], settings[f'{k}_max'])
        time_step[k] = v

    return time_step, intv_targets

@torch.no_grad()
def sample_network_data(prev_time_step, targets, settings):
    """
    Samples data from Neural Network distributions
    """
    keys = sorted(list(prev_time_step.keys()))
    if isinstance(prev_time_step[keys[0]], torch.Tensor):
        prev_inp_tensor = torch.stack([prev_time_step[key] for key in keys], dim=-1)
    else:
        prev_inp_tensor = torch.Tensor([prev_time_step[key] for key in keys])[None]
    current_inp_tensor = torch.zeros_like(prev_inp_tensor)

    for key in settings['causal_var_order']:
        i = int(key[1:])
        if targets[key] == 0 or (settings['intv_noise'] > np.random.uniform()):
            mean, std = settings['networks'][key](current_inp_tensor, prev_inp_tensor)
            current_inp_tensor[...,i] = mean + torch.randn_like(std) * std
        else:
            current_inp_tensor[...,i] = torch.randn_like(current_inp_tensor[...,i]) * settings['intv_std']
            if not settings['networks'][key].has_inputs():
                current_inp_tensor[...,i] -= 0.6

    current_inp_tensor = current_inp_tensor / 2.0

    return current_inp_tensor


class VarNet(nn.Module):

    def __init__(self, num_causal_vars, mask, default_std=-1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_causal_vars * 2, 32),
            nn.BatchNorm1d(32, momentum=0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32, momentum=0.5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, 2),
            nn.BatchNorm1d(2, momentum=0.5)       
        )
        self.default_log_std = np.log(default_std) if default_std > 0.0 else np.log(0.3)
        self.register_buffer('mask', mask)
        self.reset_parameters()

    def reset_parameters(self):
        for name, p in self.net.named_parameters():
            p.requires_grad_(False)
            if name.endswith(".bias"):
                if p.shape[0] == 2:
                    p.data.fill_(0.0)
                else:
                    p.data.uniform_(-0.2, 0.2)
            elif len(p.shape) >= 2:
                nn.init.kaiming_uniform_(p, a=0.1, nonlinearity='leaky_relu')
                target_std = p.data.std()
                if name.startswith('net.0'):
                    for i in range(p.shape[1]):
                        p[:,i] = p[:,i] / p[:,i].std() * target_std

    def forward(self, current_step, prev_step):
        if self.has_inputs():
            mask = self.mask.reshape((1,) * (len(current_step.shape)-1) + (-1,))
            inps = torch.cat([prev_step, current_step], dim=-1) * mask
            if len(inps.shape) == 1:
                inps = inps[None]
            out = self.net(inps)
            mean, log_std = out.unbind(dim=-1)
            log_std = self.default_log_std + 0.0 * torch.tanh(log_std)  # Default std 0.05, with smaller change
            std = log_std.exp()
        else:
            mean = 0.6 + prev_step.new_zeros(prev_step.shape[:-1])
            std = prev_step.new_ones(prev_step.shape[:-1])
        return mean, std

    def has_inputs(self):
        return self.mask.sum() > 0


def create_settings(seed=42, 
                    dpi=1,
                    num_causal_vars=6,
                    intv_prob=0.15,
                    intv_noise=0.0,
                    single_target_interventions=False,
                    grouped_target_interventions=False,
                    graph_type='random',
                    edge_prob_instant=0.5,
                    edge_prob_temporal=0.2,
                    num_flow_layers=2,
                    **kwargs):
    """
    Create a dictionary of the general hyperparameters for generating the Voronoi dataset.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    settings = {
        'resolution': 32,
        'dpi': dpi,
        'border_size': 2,
        'intv_std': 0.75,
        'num_causal_vars': num_causal_vars,
        'causal_var_max': np.pi*7./8,
        'causal_var_min': -np.pi*7./8,
        'intv_prob': intv_prob,
        'intv_noise': intv_noise,
        'single_target_interventions': single_target_interventions,
        'grouped_target_interventions': grouped_target_interventions,
        'graph_type': graph_type,
        'edge_prob_instant': edge_prob_instant,
        'edge_prob_temporal': edge_prob_temporal,
        'seed': seed,
        'voronoi': True,
        'num_flow_layers': num_flow_layers
    }
    settings['center_point_x'] = settings['resolution'] / 2.0
    settings['center_point_y'] = settings['resolution'] / 2.0
    for i in range(settings['num_causal_vars']):
        settings[f'c{i}_max'] = settings['causal_var_max']
        settings[f'c{i}_min'] = settings['causal_var_min']

    ######################
    ## GRAPH GENERATION ##
    ######################
    settings['causal_var_order'] = [f'c{i}' for i in range(settings['num_causal_vars'])]
    causal_order = torch.randperm(settings['num_causal_vars'])
    _, inv_causal_order = causal_order.sort()
    settings['causal_var_order'] = [settings['causal_var_order'][causal_order[i].item()] for i in range(causal_order.shape[0])]
    print('Causal var order', settings['causal_var_order'])
    probs = torch.FloatTensor(settings['num_causal_vars'], 
                              settings['num_causal_vars'])
    if settings['graph_type'] == 'random':
        graph = torch.bernoulli(probs.fill_(settings['edge_prob_instant'])).triu(diagonal=1)
    elif settings['graph_type'] == 'chain':
        graph = torch.zeros_like(probs)
        for i in range(settings['num_causal_vars']-1):
            graph[i,i+1] = 1.0
    elif settings['graph_type'] == 'full':
        graph = torch.ones_like(probs).triu(diagonal=1)
    else:
        assert False, f'Unknown graph type \"{settings["graph_type"]}\"'
    settings['causal_graph'] = graph[inv_causal_order,:][:,inv_causal_order]  # Re-ordering
    print('Causal graph', settings['causal_graph'], settings['causal_graph'].sum().item(), 'edges')
    settings['temporal_causal_graph'] = torch.bernoulli(probs.fill_(settings['edge_prob_temporal'])) # + torch.eye(settings['num_causal_vars'])
    settings['temporal_causal_graph'].clamp_(min=0.0, max=1.0)
    if settings['edge_prob_temporal'] > 0.0:
        for i in range(settings['num_causal_vars']):
            while settings['temporal_causal_graph'][:,i].sum() < (2 if settings['edge_prob_temporal']>=0.5 else 1): # settings['num_causal_vars']/2 and settings['causal_graph'][:,i].sum() == 0:
                settings['temporal_causal_graph'][:,i] = settings['temporal_causal_graph'][:,i] + torch.bernoulli(probs.fill_(settings['edge_prob_temporal']))[:,i]
                settings['temporal_causal_graph'].clamp_(min=0.0, max=1.0)
    print('Temporal causal graph', settings['temporal_causal_graph'], settings['temporal_causal_graph'].sum().item(), 'edges')
    settings['causal_graph'] = settings['causal_graph'][:settings['num_causal_vars'], :settings['num_causal_vars']]

    if settings['grouped_target_interventions']:
        code_length = math.floor(math.log2(settings['num_causal_vars'])) + 2
        settings['intervention_groups'] = [{} for _ in range(code_length)]
        for var_idx in range(settings['num_causal_vars']):
            var_code = var_idx + 1
            for g_idx in range(code_length):
                settings['intervention_groups'][g_idx][f'c{var_idx}'] = (var_code % 2)
                var_code = var_code // 2

    #############################
    ## DISTRIBUTION GENERATION ##
    #############################
    settings['networks'] = {
        f'c{i}': VarNet(settings['num_causal_vars'], mask=torch.cat([settings['temporal_causal_graph'][:,i], settings['causal_graph'][:,i]], dim=-1))
                 for i in range(settings['num_causal_vars'])
    }
    intv_targets = {key: 0 for key in settings['networks']}
    prev_time_step = {key: soft_limit(torch.randn(10000,) * settings['intv_std'], settings[f'{key}_min'], settings[f'{key}_max']) for key in settings['networks']}
    for i in range(20):
        prev_time_step = {key: soft_limit(torch.randn(10000,) * settings['intv_std'], settings[f'{key}_min'], settings[f'{key}_max']) for key in settings['networks']}
        for l in range(5):
            if l < 4:
                _ = [n.train() for n in settings['networks'].values()]
            else:
                _ = [n.eval() for n in settings['networks'].values()]
            prev_time_step = {key: torch.where(torch.rand_like(prev_time_step[key]) < settings['intv_prob'], soft_limit(torch.randn_like(prev_time_step[key]) / 2.0, settings[f'{key}_min'], settings[f'{key}_max']), prev_time_step[key]) for key in prev_time_step}
            vals = sample_network_data(prev_time_step, intv_targets, settings)
            for j in range(settings['num_causal_vars']):
                prev_time_step[f'c{j}'] = soft_limit(vals[:,j], settings[f'c{j}_min'], settings[f'c{j}_max'])
    vals = sample_network_data(prev_time_step, intv_targets, settings)
    
    ###########################
    ## NORMALIZING FLOW INIT ##
    ###########################
    settings['normalizing_flow'] = get_nf(settings)
    settings['normalizing_flow'].train()
    settings['normalizing_flow'].forward(vals)  # To init the ActNorm
    settings['normalizing_flow'].eval()

    if settings['voronoi']:
        add_voronoi(settings)

    return settings


def get_nf(settings):
    """
    Generates a randomly initialized normalizing flow for entangling the causal variables
    """
    nf = AutoregNormalizingFlow(num_vars=settings['num_causal_vars'],
                                num_flows=max(0, settings['num_flow_layers']),
                                act_fn=nn.SiLU,
                                zero_init=False,
                                use_scaling=False,
                                use_1x1_convs=True,
                                init_std_factor=0.2)
    for name, p in nf.named_parameters():
        if name.endswith('scaling'):
            p.data.fill_(-1.0)
    if settings['num_flow_layers'] == 0:
        nf.flows.append(OrthogonalFlow(settings['num_causal_vars'], LU_decomposed=False))
    nf.flows.append(ActNormFlow(settings['num_causal_vars']))
    return nf

def sample_random_point(settings):
    """
    Sample a completely independent, random point for all causal factors.
    """
    step = dict()
    for i in range(settings['num_causal_vars']):
        step[f'c{i}'] = soft_limit(torch.randn(1,) / 2.0, settings[f'c{i}_min'], settings[f'c{i}_max']).item()
    return step


def add_voronoi(settings):
    """
    Generates a random voronoi diagram that is used for the dataset
    This code has been adapted from https://gist.github.com/pv/8036995
    """

    # First step: Find appropriate center points
    np.random.seed(settings['seed'])
    too_small_dist = True
    counter = 0
    max_dist, max_res = -1, None
    while too_small_dist and counter < 1000:
        random_points = np.random.uniform(settings['border_size'] * 2, 
                                          settings['resolution'] - settings['border_size'] * 2, 
                                          size=(settings['num_causal_vars'], 2))
        distances = ((random_points[None] - random_points[:,None]) ** 2).sum(axis=-1) ** 0.5
        distances[distances == 0.0] = 1e5
        min_dist = distances.min()
        too_small_dist = (min_dist < 10)
        counter += 1
        if min_dist > max_dist:
            max_dist = min_dist
            max_res = (random_points, distances)
    random_points, distances = max_res
    
    # Second step: create voronoi
    vor = Voronoi(random_points)
    radius = settings['resolution'] * 2

    # Third step: create polygons out of voronoi diagram
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    settings['voronoi'] = {
        'regions': new_regions,
        'vertices': new_vertices
    }


def plot_matplotlib_figure(time_step, settings, filename):
    """
    Render a time step with matplotlib
    """
    # Entangle causal variables
    if settings['num_flow_layers'] >= 0:
        with torch.no_grad():
            time_step = {key: limit_inverse(torch.Tensor([time_step[key]]), settings[f'{key}_min'], settings[f'{key}_max']).item() for key in time_step}
            keys = [f'c{i}' for i in range(settings['num_causal_vars'])]
            inp = torch.Tensor([time_step[k] for k in keys])
            out = settings['normalizing_flow'](inp[None])[0].squeeze(dim=0)
            for i, key in enumerate(keys):
                time_step[key] = soft_limit(out[i] / 2.0, settings[f'{key}_min'], settings[f'{key}_max']).item()

    # Prepare figure
    fig = plt.figure(figsize=(settings['resolution'], settings['resolution']), dpi=settings['dpi'])
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.xlim(0, settings['resolution'])
    plt.ylim(0, settings['resolution'])
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    # Plot voronoi diagram
    colors = {k: hsv_to_rgb(np.array([(v + (2*np.pi if v < 0 else 0)) / (2*np.pi), 1.0, 1.0])) for k, v in time_step.items()}
    keys = [f'c{i}' for i in range(settings['num_causal_vars'])]
    vertices = np.asarray(settings['voronoi']['vertices'])
    for k, region in zip(keys, settings['voronoi']['regions']):
        polygon = vertices[region]
        plt.fill(*zip(*polygon), color=colors[k], ec=np.array([0.0, 0.0, 0.0]), linewidth=50.0)
    
    plt.savefig(filename)
    plt.close()


def export_settings(folder, settings):
    """
    Save generation settings to disk, including relevant networks.
    """
    settings = deepcopy(settings)
    nodes = [f'c{i}' for i in range(settings['num_causal_vars'])]
    visualize_graph(nodes, settings['causal_graph'])
    plt.savefig(os.path.join(folder, 'instantaneous_causal_graph.pdf'))
    plt.close()
    settings['causal_graph'] = settings['causal_graph'].numpy().tolist()

    if 'temporal_causal_graph' in settings:
        visualize_graph(nodes, settings['temporal_causal_graph'])
        plt.savefig(os.path.join(folder, 'temporal_causal_graph.pdf'))
        plt.close()
        settings['temporal_causal_graph'] = settings['temporal_causal_graph'].numpy().tolist()

    if 'networks' in settings:
        networks = settings.pop('networks')
        for key in networks:
            state_dict = networks[key].state_dict()
            torch.save(state_dict, os.path.join(folder, f'network_{key}.pt'))

    if 'normalizing_flow' in settings:
        torch.save(settings['normalizing_flow'].state_dict(), os.path.join(folder, 'nf.pt'))
        _ = settings.pop('normalizing_flow')

    with open(os.path.join(folder, 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=4)


def create_intv_dataset(num_samples, folder, seed=42, settings=None):
    """
    Generate a dataset consisting of a single sequence with num_samples data points.
    Does not include the rendering with matplotlib.
    """
    os.makedirs(folder, exist_ok=True)

    if settings is None:
        settings = create_settings(seed)
    start_point = sample_random_point(settings)
    start_point, _ = next_step(start_point, settings)
    key_list = sorted(list(start_point.keys()))
    all_steps = np.zeros((num_samples, len(key_list)), dtype=np.float32)
    all_interventions = np.zeros((num_samples-1, len(key_list)), dtype=np.float32)
    next_time_step = start_point
    for n in tqdm(range(num_samples), desc='Creating intervention dataset', leave=False):
        next_time_step, intv_targets = next_step(next_time_step, settings)
        for i, key in enumerate(key_list):
            all_steps[n, i] = next_time_step[key]
            if n > 0:
                all_interventions[n-1, i] = intv_targets[key]

    np.savez_compressed(os.path.join(folder, 'latents.npz'), 
                        latents=all_steps, 
                        targets=all_interventions, 
                        keys=key_list)
    export_settings(folder, settings)


def create_indep_dataset(num_samples, folder, seed=42, settings=None):
    """
    Generate a dataset with independent samples. Used for correlation checking.
    """
    os.makedirs(folder, exist_ok=True)

    if settings is None:
        settings = create_settings(seed)
    start_point = sample_random_point(settings)
    key_list = sorted(list(start_point.keys()))
    all_steps = np.zeros((num_samples, len(key_list)), dtype=np.float32)
    for n in range(num_samples):
        new_step = sample_random_point(settings)
        for i, key in enumerate(key_list):
            all_steps[n, i] = new_step[key]

    np.savez_compressed(os.path.join(folder, 'latents.npz'), 
                        latents=all_steps,
                        targets=np.ones_like(all_steps),
                        keys=key_list)
    export_settings(folder, settings)


def create_triplet_dataset(num_samples, folder, start_data, seed=42):
    """
    Generate a dataset for the triplet evaluation.
    """
    os.makedirs(folder, exist_ok=True)
    start_dataset = np.load(start_data)
    images = start_dataset['images']
    latents = start_dataset['latents']
    targets = start_dataset['targets']
    has_intv = (targets.sum(axis=0) > 0).astype(np.int32)

    all_latents = np.zeros((num_samples, 3, latents.shape[1]), dtype=np.float32)
    prev_images = np.zeros((num_samples, 2) + images.shape[1:], dtype=np.uint8)
    target_masks = np.zeros((num_samples, latents.shape[1]), dtype=np.uint8)
    for n in range(num_samples):
        idx1 = np.random.randint(images.shape[0])
        idx2 = np.random.randint(images.shape[0]-1)
        if idx2 >= idx1:
            idx2 += 1
        latent1 = latents[idx1]
        latent2 = latents[idx2]
        srcs = None if has_intv.sum() > 0 else np.random.randint(2, size=(latent1.shape[0],))
        while srcs is None or srcs.astype(np.float32).std() == 0.0: 
            srcs = np.random.randint(2, size=(latent1.shape[0],))
            srcs = srcs * has_intv
        latent3 = np.where(srcs == 0, latent1, latent2)
        all_latents[n,0] = latent1
        all_latents[n,1] = latent2
        all_latents[n,2] = latent3
        prev_images[n,0] = images[idx1]
        prev_images[n,1] = images[idx2]
        target_masks[n] = srcs

    np.savez_compressed(os.path.join(folder, 'latents.npz'), 
                        latents=all_latents[:,-1],
                        triplet_latents=all_latents,
                        triplet_images=prev_images,
                        triplet_targets=target_masks,
                        keys=start_dataset['keys'])


def export_figures(folder, start_index=0, end_index=-1, settings=None):
    """
    Given a numpy array of latent variables, render each data point with matplotlib.
    """
    if isinstance(folder, tuple):
        folder, start_index, end_index = folder
    latents_arr = np.load(os.path.join(folder, 'latents.npz'))
    latents = latents_arr['latents']
    keys = latents_arr['keys'].tolist()
    
    if settings is None:
        with open(os.path.join(folder, 'settings.json'), 'r') as f:
            settings = json.load(f)
        if os.path.isfile(os.path.join(folder, 'nf.pt')):
            settings['normalizing_flow'] = get_nf(settings)
            settings['normalizing_flow'].load_state_dict(torch.load(os.path.join(folder, 'nf.pt')))
            settings['normalizing_flow'].eval()

    if end_index < 0:
        end_index = latents.shape[0]

    figures = np.zeros((end_index - start_index, settings['resolution']*settings['dpi'], settings['resolution']*settings['dpi'], 3), dtype=np.uint8)
    for i in range(start_index, end_index):
        time_step = {key: latents[i,j] for j, key in enumerate(keys)}
        filename = os.path.join(folder, f'fig_{str(start_index).zfill(7)}.png')
        plot_matplotlib_figure(time_step=time_step, 
                               settings=settings, 
                               filename=filename)
        main_img = imread(filename)[:,:,:3]
        figures[i - start_index,:,:,:3] = main_img

    if start_index != 0 or end_index < latents.shape[0]:
        output_filename = os.path.join(folder, f'images_{str(start_index).zfill(8)}_{str(end_index).zfill(8)}.npz')
    else:
        output_filename = os.path.join(folder, 'images.npz')
    np.savez_compressed(output_filename,
                        imgs=figures)


def generate_full_dataset(dataset_size, folder, split_name=None, num_processes=8, independent=False, triplets=False, start_data=None, settings=None, seed=42, skip_images=False):
    """
    Generate a full dataset from latent variables to rendering with matplotlib.
    To speed up the rendering process, we parallelize it with using multiple processes.
    """
    if independent:
        create_indep_dataset(dataset_size, folder, settings=settings, seed=seed)
    elif triplets:
        create_triplet_dataset(dataset_size, folder, start_data=start_data, seed=seed)
    else:
        create_intv_dataset(dataset_size, folder, settings=settings, seed=seed)

    print(f'Starting figure export ({split_name})...')
    start_time = time.time()
    if skip_images:
        empty_images = np.zeros((dataset_size, settings['resolution']*settings['dpi'], settings['resolution']*settings['dpi'], 3), dtype=np.uint8)
        np.savez_compressed(os.path.join(folder, 'images.npz'), imgs=empty_images)
    elif num_processes > 1:
        inp_args = []
        for i in range(num_processes):
            start_index = dataset_size//num_processes*i
            end_index = dataset_size//num_processes*(i+1)
            if i == num_processes - 1:
                end_index = -1
            inp_args.append((folder, start_index, end_index))
        with Pool(num_processes) as p:
            p.map(export_figures, inp_args)
        # Merge datasets
        img_sets = sorted(glob(os.path.join(folder, 'images_*_*.npz')))
        images = np.concatenate([np.load(s)['imgs'] for s in img_sets], axis=0)
        np.savez_compressed(os.path.join(folder, 'images.npz'),
                            imgs=images)
        for s in img_sets:
            os.remove(s)
        extra_imgs = sorted(glob(os.path.join(folder, 'fig_*.png')))
        for s in extra_imgs:
            os.remove(s)
    else:
        export_figures(folder, settings=settings)

    if split_name is not None:
        images = np.load(os.path.join(folder, 'images.npz'))
        latents = np.load(os.path.join(folder, 'latents.npz'))
        if triplets:
            elem_dict = dict()
            elem_dict['latents'] = latents['triplet_latents']
            elem_dict['targets'] = latents['triplet_targets']
            elem_dict['keys'] = latents['keys']
            elem_dict['images'] = np.concatenate([latents['triplet_images'], images['imgs'][:,None]], axis=1)
        else:
            elem_dict = {key: latents[key] for key in latents.keys()}
            elem_dict['images'] = images['imgs']
        np.savez_compressed(os.path.join(folder, split_name + '.npz'),
                            **elem_dict)
        os.remove(os.path.join(folder, 'images.npz'))
        os.remove(os.path.join(folder, 'latents.npz'))

    end_time = time.time()
    dur = int(end_time - start_time)
    print(f'Finished in {dur // 60}min {dur % 60}s')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Folder to save the dataset to.')
    parser.add_argument('--dataset_size', type=int, default=150000,
                        help='Number of samples to use for the dataset.')
    parser.add_argument('--num_processes', type=int, default=8,
                        help='Number of processes to use for the rendering.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for reproducibility.')
    parser.add_argument('--dpi', type=int, default=1,
                        help='Resolution of the images is 32*dpi')
    parser.add_argument('--num_causal_vars', type=int, default=6,
                        help='Number of causal variables in the graph')
    parser.add_argument('--intv_prob', type=float, default=0.15,
                        help='Probability with which interventions should be performed')
    parser.add_argument('--intv_noise', type=float, default=0.0,
                        help='Noise in the intervention targets.')
    parser.add_argument('--single_target_interventions', action='store_true',
                        help='If True, the dataset is generated using single target interventions.')
    parser.add_argument('--grouped_target_interventions', action='store_true',
                        help='If True, the dataset is generated with minimal number of experiments.')
    parser.add_argument('--graph_type', type=str, default='random',
                        help='Which graph type to generate (random, chain, full).')
    parser.add_argument('--edge_prob_instant', type=float, default=0.5,
                        help='Probability to sample an edge between two edges in the instantaneous random graph.')
    parser.add_argument('--edge_prob_temporal', type=float, default=0.25,
                        help='Probability to sample an edge in the temporal grpah.')
    parser.add_argument('--num_flow_layers', type=int, default=2,
                        help='Number of layers to use in the normalizing flow to entangle the variables.')
    parser.add_argument('--skip_images', action='store_true',
                        help='If True, no images are generated. Used for fast debugging and investigating distributions.')
    args = parser.parse_args()
    args.output_folder = args.output_folder.replace('0.', '0')
    np.random.seed(args.seed)
    os.makedirs(args.output_folder, exist_ok=True)
    settings = create_settings(**vars(args))
    export_settings(args.output_folder, settings)

    generate_full_dataset(args.dataset_size, 
                          folder=args.output_folder, 
                          split_name='train',
                          num_processes=args.num_processes,
                          seed=args.seed,
                          settings=settings,
                          skip_images=args.skip_images)
    generate_full_dataset(args.dataset_size // 10, 
                          folder=args.output_folder, 
                          split_name='val',
                          num_processes=args.num_processes,
                          seed=args.seed,
                          settings=settings,
                          skip_images=args.skip_images)
    generate_full_dataset(args.dataset_size // 4, 
                          folder=args.output_folder, 
                          split_name='val_indep',
                          independent=True,
                          num_processes=args.num_processes,
                          seed=args.seed,
                          settings=settings,
                          skip_images=args.skip_images)
    generate_full_dataset(args.dataset_size // 10, 
                          folder=args.output_folder, 
                          split_name='val_triplets',
                          triplets=True,
                          start_data=os.path.join(args.output_folder, 'val.npz'),
                          num_processes=args.num_processes,
                          seed=args.seed,
                          settings=settings,
                          skip_images=args.skip_images)
    generate_full_dataset(args.dataset_size // 10, 
                          folder=args.output_folder, 
                          split_name='test',
                          num_processes=args.num_processes,
                          seed=args.seed,
                          settings=settings,
                          skip_images=args.skip_images)
    generate_full_dataset(args.dataset_size // 4, 
                          folder=args.output_folder, 
                          split_name='test_indep',
                          independent=True,
                          num_processes=args.num_processes,
                          seed=args.seed,
                          settings=settings,
                          skip_images=args.skip_images)
    generate_full_dataset(args.dataset_size // 10, 
                          folder=args.output_folder, 
                          split_name='test_triplets',
                          triplets=True,
                          start_data=os.path.join(args.output_folder, 'test.npz'),
                          num_processes=args.num_processes,
                          seed=args.seed,
                          settings=settings,
                          skip_images=args.skip_images)