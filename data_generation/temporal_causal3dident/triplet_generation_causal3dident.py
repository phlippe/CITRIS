"""
Create images for triplet generation for the Temporal Causal3DIdent dataset.
"""

import numpy as np
import argparse
import os
from glob import glob
import json


def create_triplet_dataset(args):
    latents = np.load(os.path.join(args.input_folder, 'latents.npy'))
    shape_latents = np.load(os.path.join(args.input_folder, 'shape_latents.npy'))
    num_vars = latents.shape[-1] + shape_latents.shape[-1]
    with open(os.path.join(args.input_folder, 'hparams.json'), 'r') as f:
        hparams = json.load(f)
    exclude_vars = hparams['exclude_vars']
    non_excl_var = [i for i in range(num_vars) if i not in exclude_vars][0]

    if args.seed < 0:
        np.random.seed(args.n_points + latents.shape[0] + num_vars)
    else:
        np.random.seed(args.seed)

    indices = np.random.randint(latents.shape[0], size=(2, args.n_points))
    if args.exclude_objects is not None:
        obj_shapes = shape_latents[:,0]
        for i in range(indices.shape[1]):
            while any([int(obj_shapes[j]) in args.exclude_objects for j in indices[:,i]]) and obj_shapes[indices[0,i]] != obj_shapes[indices[1,i]]:
                indices[1,i] = np.random.randint(latents.shape[0])
    masks = np.zeros((args.n_points, num_vars), dtype=np.int32)
    for i in range(masks.shape[0]):
        while masks[i].sum() in [0, masks.shape[1]]:
            masks[i] = np.random.randint(2, size=masks[i].shape)
            # Set excluded variables to another variable's mask so they 
            # do not influence the decision whether we need to resample
            # the mask or not.
            masks[i,exclude_vars] = masks[i,non_excl_var]
            if args.coarse_vars:
                masks[i,0:3] = masks[i,0]
                masks[i,3:6] = masks[i,3]
    print(f'Masks: {masks}')
    
    full_latents = make_triplets(latents, indices, masks[:,:latents.shape[1]])
    full_shape_latents = make_triplets(shape_latents, indices, masks[:,latents.shape[1]:])

    output_folder = args.input_folder
    if output_folder[-1] == '/':
        output_folder = output_folder[:-1]
    output_folder += f'_triplets{"_coarse" if args.coarse_vars else ""}/'
    os.makedirs(output_folder, exist_ok=True)

    # Save all data necessary for the triplets
    np.save(os.path.join(output_folder, 'full_latents.npy'), full_latents)
    np.save(os.path.join(output_folder, 'full_shape_latents.npy'), full_shape_latents)
    np.save(os.path.join(output_folder, 'sources_mask.npy'), masks)
    np.save(os.path.join(output_folder, 'sources_indices.npy'), indices)
    # Save the latents for blender to generate the new images
    np.save(os.path.join(output_folder, 'latents.npy'), full_latents[:,-1])
    np.save(os.path.join(output_folder, 'shape_latents.npy'), full_shape_latents[:,-1])

    # Copy hparams file
    hparams['orig_dataset'] = args.input_folder
    hparams['triplets'] = {
        'coarse_vars': args.coarse_vars,
        'num_triplet_points': args.n_points,
        'orig_dataset': args.input_folder
    }
    with open(os.path.join(output_folder, 'hparams.json'), 'w') as f:
        json.dump(hparams, f, indent=4)


def make_triplets(orig_arr, indices, mask):
    triplets = np.stack([orig_arr[indices[0]], orig_arr[indices[1]]], axis=0)
    new_vals = np.where(mask == 0.0, triplets[0], triplets[1])
    triplets = np.concatenate([triplets, new_vals[None]], axis=0)
    triplets = triplets.transpose(1, 0, 2)
    return triplets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', required=True, type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--coarse_vars', action='store_true')
    parser.add_argument('--exclude_objects', type=int, nargs='+', default=None)
    parser.add_argument('--seed', default=-1, type=int)
    args = parser.parse_args()
    create_triplet_dataset(args)