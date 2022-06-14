import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import matplotlib.pyplot as plt 
import seaborn as sns
import networkx as nx
import numpy as np
import os


@torch.no_grad()
def visualize_ae_reconstruction(model, images):
    """ Plots reconstructions of an autoencoder """
    reconst = model(images)
    if not isinstance(reconst, torch.Tensor):
        reconst = reconst.mean()
    images = torch.stack([images, reconst], dim=1).flatten(0, 1)
    img_grid = torchvision.utils.make_grid(images, nrow=2, normalize=True, pad_value=0.5)
    img_grid = img_grid.permute(1, 2, 0)
    img_grid = img_grid.cpu().numpy()
    fig = plt.figure()
    plt.imshow(img_grid)
    plt.axis('off')
    plt.tight_layout()
    return fig


@torch.no_grad()
def visualize_reconstruction(model, image, label, dataset):
    """ Plots the reconstructions of a VAE """
    reconst, *_ = model(image[None])
    reconst = reconst.squeeze(dim=0)

    if dataset.num_labels() > 1:
        soft_img = dataset.label_to_img(torch.softmax(reconst, dim=0))
        hard_img = dataset.label_to_img(torch.argmax(reconst, dim=0))
        if label.dtype == torch.long:
            true_img = dataset.label_to_img(label)
            diff_img = (hard_img != true_img).any(dim=-1, keepdims=True).long() * 255
        else:
            true_img = label
            soft_reconst = soft_img.float() / 255.0 * 2.0 - 1.0
            diff_img = (label - soft_reconst).clamp(min=-1, max=1)
    else:
        soft_img = reconst
        hard_img = reconst
        true_img = label
        diff_img = (label - reconst).clamp(min=-1, max=1)

    imgs = [image, true_img, soft_img, hard_img, diff_img]
    titles = ['Original image', 'GT Labels', 'Soft prediction', 'Hard prediction', 'Difference']
    imgs = [t.permute(1, 2, 0) if (t.shape[0] in [3,4] and t.shape[-1] != 3) else t for t in imgs]
    imgs = [t.detach().cpu().numpy() for t in imgs]
    imgs = [((t + 1.0) * 255.0 / 2.0).astype(np.int32) if t.dtype == np.float32 else t for t in imgs]
    imgs = [t.astype(np.uint8) for t in imgs]

    fig, axes = plt.subplots(1, len(imgs), figsize=(10, 3))
    for np_img, title, ax in zip(imgs, titles, axes):
        ax.imshow(np_img)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    return fig


@torch.no_grad()
def plot_target_assignment(prior, dataset=None):
    """ Plots the probability matrix of latent-to-causal variable assignments """
    target_probs = prior.get_target_assignment().detach().cpu().numpy()
    fig = plt.figure(figsize=(max(6, target_probs.shape[1]), max(6, target_probs.shape[0]/2.5)))
    if dataset is not None:
        target_names = dataset.target_names()
        if len(target_names) == target_probs.shape[1]-1:
            target_names = target_names + ['No variable']
    else:
        target_names = [f'Block {i+1}' for i in range(target_probs.shape[1])]
    sns.heatmap(target_probs, annot=True,
                yticklabels=[f'Dim {i+1}' for i in range(target_probs.shape[0])],
                xticklabels=target_names)
    plt.xlabel('Blocks/Causal Variable')
    plt.ylabel('Latent dimensions')
    plt.title('Soft assignment of latent variable to block')
    plt.tight_layout()
    return fig


@torch.no_grad()
def plot_target_classification(results):
    """ Plots the classification accuracies of the target classifier """
    results = {key.split('.')[-1]: results[key] for key in results.keys() if key.startswith('training_step.target_classifier')}
    if len(results) == 0:
        return None
    else:
        key_to_block = lambda key: key.split('_')[-2].replace('block','').replace('[','').replace(']','')
        key_to_class = lambda key: key.split('_class')[-1].replace('[','').replace(']','')
        blocks = sorted(list(set([key_to_block(key) for key in results])))
        classes = sorted(list(set([key_to_class(key) for key in results])))
        target_accs = np.zeros((len(blocks), len(classes)), dtype=np.float32)
        for key in results:
            target_accs[blocks.index(key_to_block(key)), classes.index(key_to_class(key))] = results[key].value / results[key].cumulated_batch_size

        fig = plt.figure(figsize=(max(4, len(classes)/1.25), max(4, len(blocks)/1.25)))
        sns.heatmap(target_accs, annot=True,
                    yticklabels=blocks,
                    xticklabels=classes)
        plt.xlabel('Variable classes/targets')
        plt.ylabel('Variable blocks')
        plt.title('Classification accuracy of blocks to causal variables')
        plt.tight_layout()
        return fig


@torch.no_grad()
def plot_latents_mutual_information(mi_estimator):
    log_matrix = mi_estimator.loss_latents_logger.detach().cpu().numpy()
    
    fig = plt.figure(figsize=(max(6, log_matrix.shape[1]), max(6, log_matrix.shape[0]/2.5)))
    sns.heatmap(log_matrix, annot=True,
                yticklabels=[f'Dim {i+1}' for i in range(log_matrix.shape[0])],
                xticklabels=[mi_estimator.var_names[i] for i in range(log_matrix.shape[1])])
    plt.xlabel('Blocks/Causal Variable')
    plt.ylabel('Latent dimensions')
    plt.title('MI estimator loss for latents per intervention')
    plt.tight_layout()
    return fig


@torch.no_grad()
def visualize_triplet_reconstruction(model, img_triplet, labels, sources, dataset=None, *args, **kwargs):
    """ Plots the triplet predictions against the ground truth for a VAE/Flow """
    sources = sources[0].to(model.device)
    labels = labels[-1]
    triplet_rec = model.triplet_prediction(img_triplet[None], sources[None])
    triplet_rec = triplet_rec.squeeze(dim=0)
    if labels.dtype == torch.long:
        triplet_rec = triplet_rec.argmax(dim=0)
        diff_img = (triplet_rec != labels).long() * 255
    else:
        diff_img = ((triplet_rec - labels).clamp(min=-1, max=1) + 1) / 2.0
    triplet_rec = dataset.label_to_img(triplet_rec)
    labels = dataset.label_to_img(labels)
    vs = [img_triplet, labels, sources, triplet_rec, diff_img]
    vs = [e.squeeze(dim=0) for e in vs]
    vs = [t.permute(0, 2, 3, 1) if (len(t.shape) == 4 and t.shape[0] in [3,4] and t.shape[-1] != 3) else t for t in vs]
    vs = [t.permute(1, 2, 0) if (len(t.shape) == 3 and t.shape[0] in [3,4] and t.shape[-1] != 3) else t for t in vs]
    vs = [e.detach().cpu().numpy() for e in vs]
    img_triplet, labels, sources, triplet_rec, diff_img = vs
    img_triplet = (img_triplet + 1.0) / 2.0
    s1 = np.where(sources == 0)[0]
    s2 = np.where(sources == 1)[0]

    fig, axes = plt.subplots(1, 5, figsize=(8, 3))
    for i, (img, title) in enumerate(zip([img_triplet[0], img_triplet[1], triplet_rec, labels, diff_img], 
                                         ['Image 1', 'Image 2', 'Reconstruction', 'GT Label', 'Difference'])):
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
    targets = dataset.target_names()
    fig.suptitle(f'Image 1: {[targets[i] for i in s1]}, Image 2: {[targets[i] for i in s2]}')
    plt.tight_layout()
    return fig


def visualize_graph(nodes, adj_matrix):
    if nodes is None:
        nodes = [f'c{i}' for i in range(adj_matrix.shape[0])]
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    edges = np.where(adj_matrix == 1)
    edges = [(nodes[edges[0][i]], nodes[edges[1][i]]) for i in range(edges[0].shape[0])]
    G.add_edges_from(edges)
    pos = nx.circular_layout(G)

    figsize = max(3, len(nodes))
    fig = plt.figure(figsize=(figsize, figsize))
    nx.draw(G, 
            pos=pos, 
            arrows=True,
            with_labels=True,
            font_weight='bold',
            node_color='lightgrey',
            edgecolors='black',
            node_size=600,
            arrowstyle='-|>',
            arrowsize=16)
    return fig