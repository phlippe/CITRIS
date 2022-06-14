"""
Create the Interventional Pong dataset including rendering.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from imageio import imread
import json
from multiprocessing import Pool
import time
from glob import glob
from argparse import ArgumentParser


def paddle_step(paddle_y, ball_y, settings, intervention=False):
    """
    Sample the next position of a paddle.
    """
    if not intervention:
        step = min(abs(paddle_y - ball_y), settings['paddle_max_step'])
        if paddle_y > ball_y:
            step *= -1
    else:
        step = settings['paddle_max_step']
        if np.random.uniform() > 0.5:
            step = -step
    new_paddle_y = paddle_y + step + np.random.randn() * settings['paddle_step_noise']
    return new_paddle_y


def sample_ball_vel_dir(settings):
    """
    Samples a ball velocity direction. Can be restricted if needed.
    """
    return np.random.uniform(0, 2*np.pi)


def mod_angle(angle):
    """
    Modulo for angles.
    """
    while angle < 0.0:
        angle += 2*np.pi
    while angle > 2*np.pi:
        angle -= 2*np.pi
    return angle


def angle_flip(angle, axis='y'):
    """
    Determines the new velocity direction when the ball hits the borders or paddles.
    """
    if axis == 'x':
        angle = mod_angle(np.pi - angle)
    elif axis == 'y':
        angle = mod_angle(angle + 0.5 * np.pi)
        angle = mod_angle(np.pi - angle)
        angle = mod_angle(angle - 0.5 * np.pi)
    return angle


def ball_collision(paddle_tag, new_time_step, prev_time_step, settings):
    """
    Check whether the ball collides with a paddle.
    """
    paddle_x, paddle_y = settings[paddle_tag+'_x'], prev_time_step[paddle_tag+'_y']
    paddle_height = settings['paddle_height']
    paddle_width = settings['paddle_width']
    ball_x_center = new_time_step['ball_x']
    ball_y = new_time_step['ball_y']
    if paddle_tag.endswith('right'):
        ball_x_outer = ball_x_center + settings['ball_radius']
    else:
        ball_x_outer = ball_x_center - settings['ball_radius']
    
    if ball_y > paddle_y + paddle_height / 2.0 or ball_y < paddle_y - paddle_height / 2.0:
        return False

    if paddle_tag.endswith('right'):
        if ball_x_center < paddle_x + paddle_width / 2.0 and ball_x_outer > paddle_x - paddle_width / 2.0:
            return True
        else:
            return False
    if paddle_tag.endswith('left'):
        if ball_x_center > paddle_x - paddle_width / 2.0 and ball_x_outer < paddle_x + paddle_width / 2.0:
            return True
        else:
            return False
    return False


def hard_limit(v, v_min, v_max):
    """
    Limit a value 'v' between v_min and v_max.
    """
    return max(min(v, v_max), v_min)


def put_in_boundaries(time_step, settings):
    """
    For all variables, make sure they are in their corresponding boundaries.
    """
    for key in time_step:
        if key == 'ball_vel_dir':
            time_step[key] = mod_angle(time_step[key])
        elif (key + '_min') in settings and (key + '_max') in settings:
            time_step[key] = hard_limit(time_step[key],
                                        v_min=settings[key+'_min'],
                                        v_max=settings[key+'_max'])
    return time_step


def next_step_interventions(prev_time_step, settings, ball_reset=False):
    """
    Sample an intervention value for every causal variable.
    """
    new_time_step = dict()

    if not ball_reset:
        new_time_step['paddle_left_y'] = paddle_step(prev_time_step['paddle_left_y'], prev_time_step['ball_y'], settings, intervention=True)
        new_time_step['paddle_right_y'] = paddle_step(prev_time_step['paddle_right_y'], prev_time_step['ball_y'], settings, intervention=True)
    else:
        new_time_step['paddle_left_y'] = np.random.uniform(settings['center_point_y']*0.25, settings['center_point_y']*1.75)
        new_time_step['paddle_right_y'] = np.random.uniform(settings['center_point_y']*0.25, settings['center_point_y']*1.75)
    new_time_step['ball_x'] = np.random.uniform(settings['ball_x_min_sample'], settings['ball_x_max_sample'])
    new_time_step['ball_y'] = np.random.uniform(settings['ball_y_min'], settings['ball_y_max'])
    new_time_step['ball_vel_dir'] = sample_ball_vel_dir(settings)
    new_time_step = put_in_boundaries(new_time_step, settings)
    return new_time_step


def next_step_regular(prev_time_step, settings):
    """
    Perform transition from one time step to the next one without any interventions.
    """
    new_time_step = dict()

    new_time_step['paddle_left_y'] = paddle_step(prev_time_step['paddle_left_y'], prev_time_step['ball_y'], settings)
    new_time_step['paddle_right_y'] = paddle_step(prev_time_step['paddle_right_y'], prev_time_step['ball_y'], settings)

    vel_y = np.cos(prev_time_step['ball_vel_dir']) * prev_time_step['ball_vel_magn']
    vel_x = np.sin(prev_time_step['ball_vel_dir']) * prev_time_step['ball_vel_magn']
    new_time_step['ball_x'] = prev_time_step['ball_x'] + vel_x
    
    point_left, point_right, ball_reset = False, False, False
    if new_time_step['ball_x'] < settings['ball_x_min_point']:
        point_right = True
        ball_reset = True
    elif new_time_step['ball_x'] > settings['ball_x_max_point']:
        point_left = True
        ball_reset = True

    if ball_reset:
        new_time_step['ball_x'] = settings['center_point_x']
        new_time_step['ball_y'] = settings['center_point_y']
        new_time_step['ball_vel_dir'] = sample_ball_vel_dir(settings)
        new_time_step['paddle_left_y'] = np.random.uniform(settings['center_point_y']*0.25, settings['center_point_y']*1.75)
        new_time_step['paddle_right_y'] = np.random.uniform(settings['center_point_y']*0.25, settings['center_point_y']*1.75)
    else:
        # Deterministic ball dynamics
        new_time_step['ball_y'] = prev_time_step['ball_y'] + vel_y
        new_time_step['ball_vel_dir'] = prev_time_step['ball_vel_dir']
        # Collisions - Wall (top / bottom)
        if new_time_step['ball_y'] > settings['ball_y_max']:
            new_time_step['ball_y'] = settings['ball_y_max'] - (new_time_step['ball_y'] - settings['ball_y_max'])
            new_time_step['ball_vel_dir'] = angle_flip(new_time_step['ball_vel_dir'], axis='x')
        elif new_time_step['ball_y'] < settings['ball_y_min']:
            new_time_step['ball_y'] = settings['ball_y_min'] - (new_time_step['ball_y'] - settings['ball_y_min'])
            new_time_step['ball_vel_dir'] = angle_flip(new_time_step['ball_vel_dir'], axis='x')
        # Collision - Paddles
        if ball_collision('paddle_left', new_time_step, prev_time_step, settings):
            new_time_step['ball_x'] = (settings['paddle_left_x'] + settings['paddle_width']/2.0) * 2 - (new_time_step['ball_x'] - settings['ball_radius']*2)
            new_time_step['ball_vel_dir'] = angle_flip(new_time_step['ball_vel_dir'], axis='y')
        elif ball_collision('paddle_right', new_time_step, prev_time_step, settings):
            new_time_step['ball_x'] = (settings['paddle_right_x'] - settings['paddle_width']/2.0) * 2 - (new_time_step['ball_x'] + settings['ball_radius']*2)
            new_time_step['ball_vel_dir'] = angle_flip(new_time_step['ball_vel_dir'], axis='y')
        # Noise addition
        new_time_step['ball_x'] += np.random.randn() * settings['ball_x_noise']
        new_time_step['ball_y'] += np.random.randn() * settings['ball_y_noise']
        new_time_step['ball_vel_dir'] += np.random.randn() * settings['ball_vel_dir_noise']

    new_time_step['score_left'] = prev_time_step['score_left'] + int(point_left)
    new_time_step['score_right'] = prev_time_step['score_right'] + int(point_right)
    if max(new_time_step['score_right'], new_time_step['score_left']) >= settings['max_points']:
        new_time_step['score_left'] = 0
        new_time_step['score_right'] = 0

    new_time_step['ball_vel_magn'] = prev_time_step['ball_vel_magn']  # Constant velocity magnitude

    new_time_step = put_in_boundaries(new_time_step, settings)
    return new_time_step, ball_reset


def next_step(prev_time_step, settings):
    """
    Combines the observational and interventional dynamics to sample a new step with respecting the intervention sample policy.
    """
    std_time_step, ball_reset = next_step_regular(prev_time_step, settings)
    intv_time_step = next_step_interventions(prev_time_step, settings, ball_reset=ball_reset)
    intv_targets = {}
    if settings['single_target_interventions']:
        keys = sorted(list(intv_time_step.keys()))
        num_vars = len(keys)
        t = np.random.randint(num_vars)
        no_int_prob = (1 - settings['intv_prob']) ** num_vars
        if np.random.uniform() < no_int_prob:
            t = -1
        intv_targets = {key: int(t == i) for i, key in enumerate(keys)}
    else:
        intv_targets = {key: int(np.random.rand() < settings['intv_prob']) for key in intv_time_step}
    for key in std_time_step:
        if key in intv_time_step and intv_targets[key] == 1:
            std_time_step[key] = intv_time_step[key]
        else:
            intv_targets[key] = 0
    return std_time_step, intv_targets


def create_settings():
    """
    Create a dictionary of the general hyperparameters for generating the Pong dataset.
    """
    border_size = 2
    settings = {
        'resolution': 32,
        'dpi': 1,
        'border_size': border_size,
        'paddle_height': 6,
        'paddle_width': 2,
        'paddle_left_x': 3 + border_size,
        'paddle_left_y_min': border_size,
        'paddle_right_y_min': border_size,
        'paddle_max_step': 1.5,
        'paddle_step_noise': 0.5,
        'ball_radius': 1.2,
        'ball_x_noise': 0.2,
        'ball_y_noise': 0.2,
        'ball_vel_dir_noise': 0.1,
        'max_points': 5,
        'intv_prob': 0.15,
        'single_target_interventions': True
    }
    settings['paddle_right_x'] = settings['resolution'] - settings['paddle_left_x']
    settings['paddle_left_y_max'] = settings['resolution'] - settings['border_size']
    settings['paddle_right_y_max'] = settings['resolution'] - settings['border_size']
    settings['ball_y_max'] = settings['resolution'] - settings['ball_radius'] - settings['border_size']
    settings['ball_y_min'] = settings['ball_radius'] + settings['border_size']
    settings['ball_x_max_point'] = settings['resolution'] - settings['ball_radius'] - border_size
    settings['ball_x_min_point'] = settings['ball_radius'] + border_size
    settings['ball_x_max'] = settings['resolution'] - border_size
    settings['ball_x_min'] = border_size
    settings['ball_x_max_sample'] = settings['paddle_right_x'] - settings['paddle_width'] / 2.0 - settings['ball_radius']
    settings['ball_x_min_sample'] = settings['paddle_left_x'] + settings['paddle_width'] / 2.0 + settings['ball_radius']
    settings['center_point_x'] = settings['resolution'] / 2.0
    settings['center_point_y'] = settings['resolution'] / 2.0
    return settings


def sample_random_point(settings):
    """
    Sample a completely independent, random point for all causal factors.
    """
    step = dict()
    step['ball_x'] = np.random.uniform(settings['ball_x_min_sample'], settings['ball_x_max_sample'])
    step['ball_y'] = np.random.uniform(settings['ball_y_min'], settings['ball_y_max'])
    step['ball_vel_magn'] = 2.0
    step['ball_vel_dir'] = sample_ball_vel_dir(settings)
    step['paddle_left_y'] = np.random.uniform(settings['paddle_left_y_min'], settings['paddle_left_y_max'])
    step['paddle_right_y'] = np.random.uniform(settings['paddle_right_y_min'], settings['paddle_right_y_max'])
    step['score_left'] = np.random.randint(settings['max_points'])
    step['score_right'] = np.random.randint(settings['max_points'])
    return step


def plot_matplotlib_figure(time_step, settings, filename):
    """
    Render a time step with matplotlib
    """
    plt.figure(figsize=(settings['resolution'], settings['resolution']), dpi=settings['dpi'])
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.xlim(0, settings['resolution'])
    plt.ylim(0, settings['resolution'])
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    color_back = np.array([0.9, 0.9, 0.9])
    background = plt.Rectangle((0, 0), settings['resolution'], settings['resolution'], fc=color_back)
    ax.add_patch(background)
    digit_color = np.array([0.5, 0.5, 0.5])
    write_digit(x=settings['resolution']/2.0-4, y=24, digit=time_step['score_left'], color=digit_color, ax=ax)
    rec = plt.Rectangle((settings['resolution']/2.0, 25), 1, 1, fc=digit_color)
    ax.add_patch(rec)
    rec = plt.Rectangle((settings['resolution']/2.0, 27), 1, 1, fc=digit_color)
    ax.add_patch(rec)
    write_digit(x=settings['resolution']/2.0+2, y=24, digit=time_step['score_right'], color=digit_color, ax=ax)
    paddle_left = plt.Rectangle((settings['paddle_left_x'] - settings['paddle_width']/2.0, time_step['paddle_left_y'] - settings['paddle_height']/2.0), settings['paddle_width'], settings['paddle_height'], fc='b', snap=False)
    ax.add_patch(paddle_left)
    paddle_right = plt.Rectangle((settings['paddle_right_x'] - settings['paddle_width']/2.0, time_step['paddle_right_y'] - settings['paddle_height']/2.0), settings['paddle_width'], settings['paddle_height'], fc='g', snap=False)
    ax.add_patch(paddle_right)
    border_left = plt.Rectangle((0, 0), settings['border_size'], settings['resolution'], fc=np.array([0.1,0.1,0.1]))
    ax.add_patch(border_left)
    border_right = plt.Rectangle((settings['resolution']-settings['border_size'], 0), settings['border_size'], settings['resolution'], fc=np.array([0.1,0.1,0.1]))
    ax.add_patch(border_right)
    border_bottom = plt.Rectangle((0, 0), settings['resolution'], settings['border_size'], fc=np.array([0.1,0.1,0.1]))
    ax.add_patch(border_bottom)
    border_top = plt.Rectangle((0, settings['resolution']-settings['border_size']), settings['resolution'], settings['border_size'], fc=np.array([0.1,0.1,0.1]))
    ax.add_patch(border_top)
    ball = plt.Circle((time_step['ball_x'], time_step['ball_y']), radius=settings['ball_radius'], fc='r')
    ax.add_patch(ball)
    
    plt.savefig(filename)
    plt.close()

    plt.figure(figsize=(settings['resolution'], settings['resolution']), dpi=settings['dpi'])
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.xlim(0, settings['resolution'])
    plt.ylim(0, settings['resolution'])
    plt.yticks([])
    plt.xticks([])
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    vel_x = np.sin(time_step['ball_vel_dir']) * time_step['ball_vel_magn']
    vel_y = np.cos(time_step['ball_vel_dir']) * time_step['ball_vel_magn']
    ball_x_proj = time_step['ball_x'] + vel_x
    ball_y_proj = time_step['ball_y'] + vel_y
    ball = plt.Circle((ball_x_proj, ball_y_proj), radius=settings['ball_radius'], fc=np.array([0.0,0.0,0.0]))
    ax.add_patch(ball)

    plt.savefig(filename.replace('.png', '_proj.png'))
    plt.close()


def write_digit(x, y, digit, color, ax):
    """
    Writing the score digits as matplotlib rectangles
    """
    if digit == 0:
        rec = plt.Rectangle((x, y), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y), 1, 5, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x+2, y), 1, 5, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+4), 3, 1, fc=color)
        ax.add_patch(rec)
    if digit == 1:
        rec = plt.Rectangle((x+1, y), 1, 5, fc=color)
        ax.add_patch(rec)
    if digit == 2:
        rec = plt.Rectangle((x, y), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+4), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x+2, y+2), 1, 3, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y), 1, 3, fc=color)
        ax.add_patch(rec)
    if digit == 3:
        rec = plt.Rectangle((x+2, y), 1, 5, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+4), 3, 1, fc=color)
        ax.add_patch(rec)
    if digit == 4:
        rec = plt.Rectangle((x+2, y), 1, 5, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 1, 3, fc=color)
        ax.add_patch(rec)
    if digit == 5:
        rec = plt.Rectangle((x, y), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+4), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x+2, y), 1, 3, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 1, 3, fc=color)
        ax.add_patch(rec)


def create_seq_dataset(num_samples, folder):
    """
    Generate a dataset consisting of a single sequence with num_samples data points.
    Does not include the rendering with matplotlib.
    """
    os.makedirs(folder, exist_ok=True)

    settings = create_settings()
    start_point = sample_random_point(settings)
    start_point, _ = next_step(start_point, settings)
    key_list = sorted(list(start_point.keys()))
    all_steps = np.zeros((num_samples, len(key_list)), dtype=np.float32)
    all_interventions = np.zeros((num_samples-1, len(key_list)), dtype=np.float32)
    next_time_step = start_point
    for n in range(num_samples):
        next_time_step, intv_targets = next_step(next_time_step, settings)
        for i, key in enumerate(key_list):
            all_steps[n, i] = next_time_step[key]
            if n > 0:
                all_interventions[n-1, i] = intv_targets[key]

    np.savez_compressed(os.path.join(folder, 'latents.npz'), 
                        latents=all_steps, 
                        targets=all_interventions, 
                        keys=key_list)
    with open(os.path.join(folder, 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=4)


def create_indep_dataset(num_samples, folder):
    """
    Generate a dataset with independent samples. Used for correlation checking.
    """
    os.makedirs(folder, exist_ok=True)

    settings = create_settings()
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
    with open(os.path.join(folder, 'settings.json'), 'w') as f:
        json.dump(settings, f, indent=4)


def create_triplet_dataset(num_samples, folder, start_data):
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
        # Pick two random images that we want to combine
        idx1 = np.random.randint(images.shape[0])
        idx2 = np.random.randint(images.shape[0]-1)
        if idx2 >= idx1:
            idx2 += 1
        latent1 = latents[idx1]
        latent2 = latents[idx2]
        # Pick a random combination of both images for a new, third image
        srcs = None if has_intv.sum() > 0 else np.random.randint(2, size=(latent1.shape[0],))
        while srcs is None or srcs.astype(np.float32).std() == 0.0:  
            # Prevent that we take all causal variables from one of the two images
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


def export_figures(folder, start_index=0, end_index=-1):
    """
    Given a numpy array of latent variables, render each data point with matplotlib.
    """
    if isinstance(folder, tuple):
        folder, start_index, end_index = folder
    latents_arr = np.load(os.path.join(folder, 'latents.npz'))
    latents = latents_arr['latents']
    keys = latents_arr['keys'].tolist()
    
    with open(os.path.join(folder, 'settings.json'), 'r') as f:
        settings = json.load(f)

    if end_index < 0:
        end_index = latents.shape[0]

    figures = np.zeros((end_index - start_index, settings['resolution']*settings['dpi'], settings['resolution']*settings['dpi'], 4), dtype=np.uint8)
    for i in range(start_index, end_index):
        time_step = {key: latents[i,j] for j, key in enumerate(keys)}
        filename = os.path.join(folder, f'fig_{str(start_index).zfill(7)}.png')
        plot_matplotlib_figure(time_step=time_step, 
                               settings=settings, 
                               filename=filename)
        main_img = imread(filename)[:,:,:3]
        move_img = imread(filename.replace('.png', '_proj.png'))[:,:,:1]
        figures[i - start_index,:,:,:3] = main_img
        figures[i - start_index,:,:,3:] = move_img

    if start_index != 0 or end_index < latents.shape[0]:
        output_filename = os.path.join(folder, f'images_{str(start_index).zfill(8)}_{str(end_index).zfill(8)}.npz')
    else:
        output_filename = os.path.join(folder, 'images.npz')
    np.savez_compressed(output_filename,
                        imgs=figures)


def generate_full_dataset(dataset_size, folder, split_name=None, num_processes=8, independent=False, triplets=False, start_data=None):
    """
    Generate a full dataset from latent variables to rendering with matplotlib.
    To speed up the rendering process, we parallelize it with using multiple processes.
    """
    if independent:
        create_indep_dataset(dataset_size, folder)
    elif triplets:
        create_triplet_dataset(dataset_size, folder, start_data=start_data)
    else:
        create_seq_dataset(dataset_size, folder)

    print('Starting figure export...')
    start_time = time.time()
    if num_processes > 1:
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
        export_figures(folder)

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
    parser.add_argument('--dataset_size', type=int, default=100000,
                        help='Number of samples to use for the dataset.')
    parser.add_argument('--num_processes', type=int, default=8,
                        help='Number of processes to use for the rendering.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for reproducibility.')
    args = parser.parse_args()
    np.random.seed(args.seed)

    generate_full_dataset(args.dataset_size, 
                          folder=args.output_folder, 
                          split_name='train',
                          num_processes=args.num_processes)
    generate_full_dataset(args.dataset_size // 10, 
                          folder=args.output_folder, 
                          split_name='val',
                          num_processes=args.num_processes)
    generate_full_dataset(args.dataset_size // 4, 
                          folder=args.output_folder, 
                          split_name='val_indep',
                          independent=True,
                          num_processes=args.num_processes)
    generate_full_dataset(args.dataset_size // 10, 
                          folder=args.output_folder, 
                          split_name='val_triplets',
                          triplets=True,
                          start_data=os.path.join(args.output_folder, 'val.npz'),
                          num_processes=args.num_processes)
    generate_full_dataset(args.dataset_size // 10, 
                          folder=args.output_folder, 
                          split_name='test',
                          num_processes=args.num_processes)
    generate_full_dataset(args.dataset_size // 4, 
                          folder=args.output_folder, 
                          split_name='test_indep',
                          independent=True,
                          num_processes=args.num_processes)
    generate_full_dataset(args.dataset_size // 10, 
                          folder=args.output_folder, 
                          split_name='test_triplets',
                          triplets=True,
                          start_data=os.path.join(args.output_folder, 'test.npz'),
                          num_processes=args.num_processes)