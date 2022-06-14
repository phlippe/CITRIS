"""
Create the Causal Pinball dataset including rendering.
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
from copy import copy, deepcopy
import torch
import torch.nn as nn

import sys
sys.path.append('../')

from models.shared import visualize_graph


def check_for_collisions(time_step, objects, settings):
    """
    Checks for any collision in the path of the ball movement
    """
    collision_objects = []
    x = time_step['ball_x'] + np.linspace(0, time_step['ball_x_vel'], num=200) * settings['speed_mult']
    x_min = x - settings['ball_radius']
    x_max = x + settings['ball_radius']
    y = time_step['ball_y'] + np.linspace(0, time_step['ball_y_vel'], num=200) * settings['speed_mult']
    y_min = y - settings['ball_radius']
    y_max = y + settings['ball_radius']
    for key in objects:
        obj = objects[key]
        if obj['shape'] == 'rectangle':
            coll_right = (obj['xmax'] > x_min) & (obj['xmax'] < x_max) & (obj['ymin'] <= y) & (obj['ymax'] >= y)
            coll_left = (obj['xmin'] > x_min) & (obj['xmin'] < x_max) & (obj['ymin'] <= y) & (obj['ymax'] >= y)
            coll_top = (obj['ymax'] > y_min) & (obj['ymax'] < y_max) & (obj['xmin'] <= x) & (obj['xmax'] >= x)
            coll_bottom = (obj['ymin'] > y_min) & (obj['ymin'] < y_max) & (obj['xmin'] <= x) & (obj['xmax'] >= x)
            corners = np.array([
                [obj['xmin'], obj['ymin']],
                [obj['xmax'], obj['ymin']],
                [obj['xmin'], obj['ymax']],
                [obj['xmax'], obj['ymax']]
            ])
            dist_corners = ((x[:,None] - corners[None,:,0]) ** 2 + (y[:,None] - corners[None,:,1]) ** 2) ** 0.5
            coll_corners = (dist_corners < settings['ball_radius'])
            if coll_right.any() or coll_left.any() or coll_top.any() or coll_bottom.any() or coll_corners.any():
                time_step = np.where(coll_right | coll_left | coll_top | coll_bottom | coll_corners.any(axis=-1))[0][0]
                collision_objects.append({'object': obj, 'key': key, 'time_step': time_step, 
                                          'coll_top': coll_top, 'coll_bottom': coll_bottom, 
                                          'coll_right': coll_right, 'coll_left': coll_left,
                                          'coll_corners': coll_corners})
        elif obj['shape'] == 'circle':
            dist = ((x - obj['x']) ** 2 + (y - obj['y']) ** 2) ** 0.5
            coll = dist < (settings['ball_radius'] + obj['radius'])
            if coll.any():
                time_step = np.where(coll)[0][0]
                collision_objects.append({'object': obj, 'key': key, 'time_step': time_step, 'dist': dist, 'coll': coll})
        elif obj['shape'] == 'triangle':
            x_triag = np.linspace(obj['points'][0,0], obj['points'][-1,0], num=100) 
            y_triag = np.linspace(obj['points'][0,1], obj['points'][-1,1], num=100) 
            dist = ((x[:,None] - x_triag[None,:]) ** 2 + (y[:,None] - y_triag[None,:]) ** 2) ** 0.5
            coll = dist < settings['ball_radius']
            if coll.any():
                time_step = np.where(coll.any(axis=-1))[0][0]
                collision_objects.append({'object': obj, 'key': key, 'time_step': time_step, 'dist': dist, 'coll': coll})
                
    return collision_objects, {'x': x, 'y': y}


def update_velocity_from_collision(time_step, collision):
    """
    Given a collision with an object, determine the next velocity orientation and magnitude of the ball
    """
    obj = collision['object']
    time_point = collision['time_step']
    in_vec = np.array([time_step['ball_x_vel'], time_step['ball_y_vel']])
    if obj['shape'] == 'rectangle':
        if collision['coll_right'][time_point]:
            normal = np.array([1, 0])
        elif collision['coll_left'][time_point]:
            normal = np.array([-1, 0])
        elif collision['coll_top'][time_point]:
            normal = np.array([0, 1])
        elif collision['coll_bottom'][time_point]:
            normal = np.array([0, -1])
        elif collision['coll_corners'][time_point, 0]:
            normal = np.array([-1, -1])
        elif collision['coll_corners'][time_point, 1]:
            normal = np.array([1, -1])
        elif collision['coll_corners'][time_point, 2]:
            normal = np.array([-1, 1])
        elif collision['coll_corners'][time_point, 3]:
            normal = np.array([1, 1])
    elif obj['shape'] == 'circle':
        normal = np.array([time_step['ball_x'] - obj['x'], time_step['ball_y'] - obj['y']])
    elif obj['shape'] == 'triangle':
        normal = obj['points'][0]-obj['points'][-1]
        # Turn by 90 degree
        normal[0] *= -1
        normal = normal[::-1]
        if normal[-1] < 0:
            normal *= -1 # wrong orientation of the normal, quick hack!
    normal = normal / np.linalg.norm(normal)
    if np.dot(in_vec, normal) > 0:
        out_vec = normal * np.linalg.norm(in_vec)
    else:
        out_vec = in_vec - 2 * (np.dot(in_vec, normal)) * normal
    time_step['ball_x_vel'] = out_vec[0] * 0.9
    time_step['ball_y_vel'] = out_vec[1] * (0.3 if collision['key'].startswith('paddle') else 0.9)
    return time_step
        

def perform_collisions(time_step, settings):
    """
    Check for collisions of the ball in the next time step, and determine next position of ball
    """
    objects = settings['static_objects']
    for key in objects:
        if f'{key}_y_pos' in time_step:
            objects[key]['ymin'] = 0
            objects[key]['ymax'] = time_step[f'{key}_y_pos']
    
    time_step_left = copy(time_step)
    time_step = copy(time_step)
    time_done = 0
    obj_keys = []
    iter_counter = 0
    while time_done < 0.98:
        colliding_objects, own_obj = check_for_collisions(time_step_left, objects, settings)
        if len(colliding_objects) == 0:
            time_step['ball_x'] += time_step_left['ball_x_vel'] * settings['speed_mult']
            time_step['ball_y'] += time_step_left['ball_y_vel'] * settings['speed_mult']
            time_done = 1
        else:
            coll_time_point = min([o['time_step'] for o in colliding_objects])
            closest_obj = [o for o in colliding_objects if o['time_step'] == coll_time_point]
            if len(closest_obj) > 1:
                idxs = []
                for o in closest_obj:
                    if o['object']['shape'] == 'rectangle':
                        i = 2 * sum([float(o[key][coll_time_point]) for key in o if (key.startswith('coll') and key != 'coll_corners')])
                        idxs.append(i)
                    else:
                        idxs.append(0)
                closest_obj = closest_obj[idxs.index(max(idxs))]
            else:
                closest_obj = closest_obj[0]
            obj_keys.append(closest_obj['key'])
            if time_step['ball_x_vel'] == 0.0 and time_step['ball_y_vel'] == 0.0:
                break
            time_diff = (1 - time_done) * (coll_time_point / 199.0)
            time_done += time_diff
            time_step['ball_x'] = own_obj['x'][coll_time_point]
            time_step['ball_y'] = own_obj['y'][coll_time_point]
            time_step = update_velocity_from_collision(time_step, closest_obj)
            if time_step is None:
                print('Something went wrong')
                break
            # Push away from colliding object
            time_step['ball_x'] += (1 - time_done + time_diff) * 0.05 * time_step['ball_x_vel'] * settings['speed_mult']
            time_step['ball_y'] += (1 - time_done + time_diff) * 0.05 * time_step['ball_y_vel'] * settings['speed_mult']
            time_step_left['ball_x'] = time_step['ball_x']
            time_step_left['ball_y'] = time_step['ball_y']
            time_step_left['ball_x_vel'] = (1 - time_done) * time_step['ball_x_vel']
            time_step_left['ball_y_vel'] = (1 - time_done) * time_step['ball_y_vel']
        iter_counter += 1
        if iter_counter > 100:
            break
    return time_step, obj_keys


def full_step_dynamics(time_step, interventions, settings):
    """
    Perform the full game dynamics from time step t to t+1, including possible interventions
    """
    stats = {}
    next_time_step = copy(time_step)

    # Paddle dynamics
    paddle_keys = ['paddle_left', 'paddle_right']
    for paddle_key in paddle_keys:
        pos_key = f'{paddle_key}_y_pos'
        obj = settings['static_objects'][paddle_key]
        if not interventions[paddle_key]:
            if next_time_step[pos_key] > settings['paddle_y_pos_min']:
                next_time_step[pos_key] = max(settings['paddle_y_pos_min'], next_time_step[pos_key] - np.random.uniform(0.5, 2))
        else:
            next_time_step[pos_key] = np.random.uniform(0.5 * (settings['paddle_y_pos_min'] + settings['paddle_y_pos_max']), 
                                                        settings['paddle_y_pos_max'])
    # Paddle <-> Ball interaction
    if any([interventions[k] for k in paddle_keys]):
        if all([interventions[k] for k in paddle_keys]):
            paddle_pos = [next_time_step[f'{k}_y_pos'] for k in paddle_keys]
            paddle_key = paddle_keys[np.argmax(np.array(paddle_pos))]
        else:
            paddle_key = [k for k in paddle_keys if interventions[k]][0]
        pos_key = f'{paddle_key}_y_pos'
        obj = settings['static_objects'][paddle_key]
        max_vel = settings['paddle_y_pos_max'] - settings['paddle_y_pos_min']
        vel = np.random.uniform(0.25*max_vel, max_vel) # next_time_step[pos_key] - settings['paddle_y_pos_min'] # time_step[pos_key]
        if (next_time_step['ball_y'] - settings['ball_radius']) < settings['paddle_y_pos_max']:
            if next_time_step['ball_x'] >= obj['xmin'] and next_time_step['ball_x'] <= obj['xmax']:
                next_time_step['ball_y'] = settings['paddle_y_pos_max'] + settings['ball_radius'] * 2
                next_time_step['ball_y_vel'] = vel
            else:
                ys = np.linspace(0, next_time_step[pos_key], num=100)
                left_corner = (((next_time_step['ball_x'] - obj['xmin']) ** 2 + (next_time_step['ball_y'] - ys) ** 2) ** 0.5) < settings['ball_radius']
                right_corner = (((next_time_step['ball_x'] - obj['xmax']) ** 2 + (next_time_step['ball_y'] - ys) ** 2) ** 0.5) < settings['ball_radius']
                hit_left = left_corner.any()
                hit_right = right_corner.any()
                if hit_left or hit_right:
                    next_time_step['ball_y'] = settings['paddle_y_pos_max'] + settings['ball_radius'] * 1.1
                    next_time_step['ball_x'] += (next_time_step['ball_y'] - time_step['ball_y']) * (-1 if hit_left else 1)
                    next_time_step['ball_y_vel'] = vel * (2 ** 0.5)
                    next_time_step['ball_x_vel'] += (vel * (2 ** 0.5)) * (-1 if hit_left else 1)
            stats[f'{paddle_key}_hit'] = {'vel': vel}
    # Ball movement
    if interventions['ball'] or next_time_step['ball_x'] < (settings['paddle_y_pos_min'] - settings['paddle_height']):
        x_space = settings['paddle_width'] - settings['ball_radius'] * 1.5
        next_time_step['ball_x'] = settings['resolution']/2 + np.random.uniform(-x_space, x_space)
        next_time_step['ball_y'] = settings['paddle_y_pos_max'] + settings['ball_radius'] + np.random.uniform(0, 2)
        next_time_step['ball_x_vel'] = np.random.randn(1,)[0] * 0.5
        next_time_step['ball_y_vel'] = np.random.randn(1,)[0] * 0.5 - 0.5
        obj_collisions = []
    else:
        next_time_step, obj_collisions = perform_collisions(next_time_step, settings)
        next_time_step['ball_x_vel'] *= 0.98
        next_time_step['ball_y_vel'] = (next_time_step['ball_y_vel'] - 0.1) * 0.98
        assert next_time_step['ball_x'] > 0 and next_time_step['ball_x'] < settings['resolution'] and \
               next_time_step['ball_y'] > 0 and next_time_step['ball_y'] < settings['resolution'], 'Something went wrong in the collisions'
        stats['collisions'] = obj_collisions
    # Bumpers + Score dynamics
    cylinders = [k for k in settings['static_objects'] if k.startswith('cyl_')]
    for key in cylinders:
        act_key = f'{key}_active'
        if interventions['cyl']:
            next_time_step[act_key] = float(np.random.uniform() > 0.5)
        elif key in obj_collisions:
            next_time_step[act_key] = 1.0
        else:
            next_time_step[act_key] = max(0.0, next_time_step[act_key] - 0.1)
        next_time_step['score'] = (int(next_time_step[act_key] == 1) * (np.random.uniform() > 0.05) + next_time_step['score']) % settings['max_score']
            
    if interventions['score']:
        next_time_step['score'] = int(np.random.randint(low=0, high=5))
    
    return next_time_step, stats


def next_step(time_step, settings):
    """
    Sample a new time step t+1 from t, with randomly sampled intervention targets.
    """
    interventions = {}
    for key in settings['intv_probs']:
        interventions[key] = (np.random.uniform() < settings['intv_probs'][key])
    next_time_step, _ = full_step_dynamics(time_step, interventions, settings)
    return next_time_step, interventions


def create_settings(seed=42):
    """
    Create a dictionary of the general hyperparameters for generating the Causal Pinball dataset.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    settings = {
        'resolution': 32,
        'dpi': 2,
        'border_size_vertical': 2,
        'border_size_horizontal': 2,
        'triangle_border_size': 5,
        'triangle_border_offset': 7,
        'triangle_width_factor': 0.35,
        'paddle_y_pos_min': 2,
        'paddle_y_pos_max': 6,
        'paddle_height': 1,
        'cyl_radius': 1.5,
        'cyl_y_center': 21,
        'cyl_distance': 5,
        'ball_radius': 1,
        'speed_mult': 1.0,
        'max_score': 20,
        'intv_probs': {
            'ball': 0.1,
            'cyl': 0.05,
            'paddle_left': 0.2,
            'paddle_right': 0.2,
            'score': 0.05
        }
    }
    settings['triangle_width'] = settings['triangle_width_factor'] * settings['resolution'] - settings['border_size_vertical']
    settings['paddle_width'] = settings['resolution'] / 2 - (settings['triangle_width'] + settings['border_size_vertical'])
    settings['keys_targets'] = sorted(list(settings['intv_probs'].keys()))

    static_objects = {}
    static_objects['border_left'] = {
        'shape': 'rectangle',
        'xmin': 0,
        'xmax': settings['border_size_vertical'],
        'ymin': 0,
        'ymax': settings['resolution']
    }
    static_objects['border_right'] = {
        'shape': 'rectangle',
        'xmin': settings['resolution']-settings['border_size_vertical'],
        'xmax': settings['resolution'],
        'ymin': 0,
        'ymax': settings['resolution']
    }
    static_objects['border_top'] = {
        'shape': 'rectangle',
        'xmin': 0,
        'xmax': settings['resolution'],
        'ymin': settings['resolution']-settings['border_size_horizontal'],
        'ymax': settings['resolution']
    }
    static_objects['border_bottom_left'] = {
        'shape': 'rectangle',
        'xmin': settings['border_size_vertical'],
        'xmax': settings['triangle_width'] + settings['border_size_vertical'],
        'ymin': 0,
        'ymax': settings['triangle_border_offset']
    }
    static_objects['border_bottom_right'] = {
        'shape': 'rectangle',
        'xmin': settings['resolution'] - settings['border_size_vertical'] - settings['triangle_width'],
        'xmax': settings['resolution'] - settings['border_size_vertical'],
        'ymin': 0,
        'ymax': settings['triangle_border_offset']
    }
    static_objects['triangle_left'] = {
        'shape': 'triangle',
        'points': np.array([
            [settings['border_size_vertical'], settings['triangle_border_offset'] + settings['triangle_border_size']],
            [settings['border_size_vertical'], settings['triangle_border_offset']],
            [settings['border_size_vertical'] + settings['triangle_width'], settings['triangle_border_offset']]
        ])
    }
    static_objects['triangle_right'] = {
        'shape': 'triangle',
        'points': np.array([
            [settings['resolution'] - settings['border_size_vertical'], settings['triangle_border_offset'] + settings['triangle_border_size']],
            [settings['resolution'] - settings['border_size_vertical'], settings['triangle_border_offset']],
            [settings['resolution'] - settings['border_size_vertical'] - settings['triangle_width'], settings['triangle_border_offset']]
        ])
    }
    # cyl for points
    cyl_idx = 0
    for row in range(2):
        for col in range(2):
            for center in range(2 if (row + col == 0) else 1):
                x = settings['resolution']*0.5 + (1 - center) * settings['cyl_distance'] * (2 * col - 1)
                y = settings['cyl_y_center'] + (1 - center) * settings['cyl_distance'] * (2 * row - 1)
                static_objects[f'cyl_{cyl_idx}'] = {
                    'shape': 'circle',
                    'x': x,
                    'y': y,
                    'radius': settings['cyl_radius']
                }
                cyl_idx += 1
    static_objects['paddle_left'] = {
        'shape': 'rectangle',
        'xmin': settings['resolution']/2-settings['paddle_width'],
        'xmax': settings['resolution']/2,
        'height': settings['paddle_height']
    }
    static_objects['paddle_right'] = {
        'shape': 'rectangle',
        'xmin': settings['resolution']/2,
        'xmax': settings['resolution']/2+settings['paddle_width'],
        'height': settings['paddle_height']
    }

    settings['static_objects'] = static_objects

    return settings


def sample_random_point(settings):
    """
    Sample a completely independent, random point for all causal factors.
    """
    collision = True
    time_step = {
        'ball_x_vel': 0.0,
        'ball_y_vel': 0.0,
        'paddle_left_y_pos': np.random.uniform(settings['paddle_y_pos_min'], settings['paddle_y_pos_max']),
        'paddle_right_y_pos': np.random.randint(settings['paddle_y_pos_min'], settings['paddle_y_pos_max']),
        'score': np.random.randint(low=0, high=settings['max_score'])
    }
    for i in range(5):
        time_step[f'cyl_{i}_active'] = float(np.random.randint(low=0, high=11)) / 10

    objects = settings['static_objects']
    for key in objects:
        if f'{key}_y_pos' in time_step:
            objects[key]['ymin'] = 0
            objects[key]['ymax'] = time_step[f'{key}_y_pos']
    
    x_gap = settings['border_size_vertical'] + settings['ball_radius']
    while collision:
        time_step['ball_x'] = np.random.uniform(x_gap, settings['resolution'] - x_gap)
        time_step['ball_y'] = np.random.uniform(settings['paddle_y_pos_max'] + settings['ball_radius'], settings['resolution'] - settings['border_size_horizontal'] - settings['ball_radius'])
        coll_objs, _ = check_for_collisions(time_step, objects, settings)
        collision = len(coll_objs) > 0
    time_step['ball_x_vel'] = np.random.uniform(-3, 3)
    time_step['ball_y_vel'] = np.random.uniform(-3, 3)
    return time_step


def plot_matplotlib_figure(time_step, settings, filename=None, ignore_projection=False):
    """
    Render a time step with matplotlib
    """
    if filename is not None:
        fig = plt.figure(figsize=(settings['resolution'], settings['resolution']), dpi=settings['dpi'])
    else:
        fig = plt.figure(figsize=(5, 5), dpi=settings['dpi'])
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

    for key in settings['static_objects']:
        obj = settings['static_objects'][key]
        if key.startswith('border'):
            border = plt.Rectangle((obj['xmin'], obj['ymin']), obj['xmax']-obj['xmin'], obj['ymax']-obj['ymin'], fc=np.array([0.1,0.1,0.1]))
            ax.add_patch(border)
        elif key.startswith('triangle'):
            triag = plt.Polygon(obj['points'], fc=np.array([0.1,0.1,0.1]))
            ax.add_patch(triag)
        elif key.startswith('cyl_'):
            outer = plt.Circle((obj['x'], obj['y']), radius=obj['radius'], fc=np.array([0.5, 0.5, 0.5]))
            inner_color = 0.2 + 0.7 * (1 - time_step[f'{key}_active'])
            inner = plt.Circle((obj['x'], obj['y']), radius=obj['radius']/2, fc=np.array([1.0, inner_color, inner_color]))
            ax.add_patch(outer)
            ax.add_patch(inner)
        elif key.startswith('paddle_'):
            paddle = plt.Rectangle((obj['xmin'], time_step[f'{key}_y_pos'] - obj['height']), obj['xmax']-obj['xmin'], obj['height'],
                                   fc=np.array([0.5, 0.5, 0.5]))
            ax.add_patch(paddle)
    
    ball = plt.Circle((time_step['ball_x'], time_step['ball_y']), radius=settings['ball_radius'], fc=np.array([0.7, 0.7, 0.7]))
    ax.add_patch(ball)
    obj = settings['static_objects']['border_bottom_right']
    point_patch = plt.Rectangle((obj['xmin'] + 1, obj['ymin'] + 1), obj['xmax']-obj['xmin']-2, obj['ymax']-obj['ymin']-2, fc=np.array([0.8, 0.8, 0.8]))
    ax.add_patch(point_patch)
    write_digit(obj['xmin']+1, obj['ymin'] + 1, digit=int(time_step['score'] // 10), color=np.array([0.4, 0.4, 0.9]), ax=ax)
    write_digit(obj['xmin']+5, obj['ymin'] + 1, digit=time_step['score'] % 10, color=np.array([0.4, 0.4, 0.9]), ax=ax)
    
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

    if filename is not None and not ignore_projection:
        # Velocity visualization
        plt.figure(figsize=(settings['resolution'], settings['resolution']), dpi=settings['dpi'])
        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        plt.xlim(0, settings['resolution'])
        plt.ylim(0, settings['resolution'])
        plt.yticks([])
        plt.xticks([])
        plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

        ball_x_proj = settings['resolution'] / 2 + 2 * time_step['ball_x_vel']
        ball_y_proj = settings['resolution'] / 2 + 2 * time_step['ball_y_vel']
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
    elif digit == 1:
        rec = plt.Rectangle((x+1, y), 1, 5, fc=color)
        ax.add_patch(rec)
    elif digit == 2:
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
    elif digit == 3:
        rec = plt.Rectangle((x+2, y), 1, 5, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+4), 3, 1, fc=color)
        ax.add_patch(rec)
    elif digit == 4:
        rec = plt.Rectangle((x+2, y), 1, 5, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 1, 3, fc=color)
        ax.add_patch(rec)
    elif digit == 5:
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
    elif digit == 6:
        rec = plt.Rectangle((x, y), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+4), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x+2, y), 1, 3, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y), 1, 5, fc=color)
        ax.add_patch(rec)
    elif digit == 7:
        rec = plt.Rectangle((x+2, y), 1, 5, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+4), 3, 1, fc=color)
        ax.add_patch(rec)
    elif digit == 8:
        rec = plt.Rectangle((x, y), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+4), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x+2, y), 1, 5, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y), 1, 5, fc=color)
        ax.add_patch(rec)
    elif digit == 9:
        rec = plt.Rectangle((x, y), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+4), 3, 1, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x+2, y), 1, 5, fc=color)
        ax.add_patch(rec)
        rec = plt.Rectangle((x, y+2), 1, 3, fc=color)
        ax.add_patch(rec)

def export_settings(folder, settings):
    settings = deepcopy(settings)
    _ = settings.pop('static_objects')
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
    start_point, intv_targets = next_step(start_point, settings)
    key_list_latents = sorted(list(start_point.keys()))
    key_list_targets = settings['keys_targets']
    all_steps = np.zeros((num_samples, len(key_list_latents)), dtype=np.float32)
    all_interventions = np.zeros((num_samples-1, len(key_list_targets)), dtype=np.float32)
    next_time_step = start_point
    for n in tqdm(range(num_samples), leave=False, desc='Generating data points'):
        next_time_step, intv_targets = next_step(next_time_step, settings)
        for i, key in enumerate(key_list_latents):
            all_steps[n, i] = next_time_step[key]
        if n > 0:
            for i, key in enumerate(key_list_targets):
                all_interventions[n-1, i] = intv_targets[key]

    np.savez_compressed(os.path.join(folder, 'latents.npz'), 
                        latents=all_steps, 
                        targets=all_interventions, 
                        keys=key_list_latents,
                        keys_targets=key_list_targets)
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
    targets = np.ones((num_samples, len(settings['intv_probs'])), dtype=np.float32)

    np.savez_compressed(os.path.join(folder, 'latents.npz'), 
                        latents=all_steps,
                        targets=targets,
                        keys=key_list,
                        keys_targets=settings['keys_targets'])
    export_settings(folder, settings)


def create_triplet_dataset(num_samples, folder, start_data, settings, seed=42):
    """
    Generate a dataset for the triplet evaluation.
    """
    os.makedirs(folder, exist_ok=True)
    start_dataset = np.load(start_data)
    images = start_dataset['images']
    latents = start_dataset['latents']
    targets = start_dataset['targets']
    has_intv = (targets.sum(axis=0) > 0).astype(np.int32)
    keys_latents = start_dataset['keys']
    keys_targets = sorted(list(settings['intv_probs'].keys()))
    intv_to_latents = np.array([[float(k_latent.startswith(k_target)) for k_latent in keys_latents] for k_target in keys_targets])
    assert (intv_to_latents.sum(axis=0) == 1).all(), 'Interventions and latents do not fit: ' + str(intv_to_latents)
    
    all_latents = np.zeros((num_samples, 3, latents.shape[1]), dtype=np.float32)
    prev_images = np.zeros((num_samples, 2) + images.shape[1:], dtype=np.uint8)
    target_masks = np.zeros((num_samples, intv_to_latents.shape[0]), dtype=np.uint8)
    for n in range(num_samples):
        idx1 = np.random.randint(images.shape[0])
        idx2 = np.random.randint(images.shape[0]-1)
        if idx2 >= idx1:
            idx2 += 1
        latent1 = latents[idx1]
        latent2 = latents[idx2]
        srcs = None if has_intv.sum() > 0 else np.random.randint(2, size=(intv_to_latents.shape[0],))
        while srcs is None or srcs.astype(np.float32).std() == 0.0: 
            srcs = np.random.randint(2, size=(intv_to_latents.shape[0],))
            srcs = srcs * has_intv
        srcs_latents = (srcs[:,None] * intv_to_latents).sum(axis=0)  # Map sources to latents
        latent3 = np.where(srcs_latents == 0, latent1, latent2)
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
                        keys=start_dataset['keys'],
                        keys_targets=settings['keys_targets'])


def export_figures(folder, start_index=0, end_index=-1):
    """
    Given a numpy array of latent variables, render each data point with matplotlib.
    """
    if isinstance(folder, tuple):
        folder, start_index, end_index = folder
    latents_arr = np.load(os.path.join(folder, 'latents.npz'))
    latents = latents_arr['latents']
    keys = latents_arr['keys'].tolist()
    
    settings = create_settings()

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


def generate_full_dataset(dataset_size, folder, split_name=None, num_processes=8, independent=False, triplets=False, start_data=None, settings=None, seed=42):
    """
    Generate a full dataset from latent variables to rendering with matplotlib.
    To speed up the rendering process, we parallelize it with using multiple processes.
    """
    if independent:
        create_indep_dataset(dataset_size, folder, settings=settings, seed=seed)
    elif triplets:
        create_triplet_dataset(dataset_size, folder, start_data=start_data, settings=settings, seed=seed)
    else:
        create_intv_dataset(dataset_size, folder, settings=settings, seed=seed)

    print(f'Starting figure export ({split_name})...')
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
    parser.add_argument('--dataset_size', type=int, default=150000,
                        help='Number of samples to use for the dataset.')
    parser.add_argument('--num_processes', type=int, default=8,
                        help='Number of processes to use for the rendering.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for reproducibility.')
    args = parser.parse_args()
    np.random.seed(args.seed)
    settings = create_settings(args.seed)
    os.makedirs(args.output_folder, exist_ok=True)
    export_settings(args.output_folder, settings)

    generate_full_dataset(args.dataset_size, 
                          folder=args.output_folder, 
                          split_name='train',
                          num_processes=args.num_processes,
                          seed=args.seed,
                          settings=settings)
    generate_full_dataset(args.dataset_size // 10, 
                          folder=args.output_folder, 
                          split_name='val',
                          num_processes=args.num_processes,
                          seed=args.seed,
                          settings=settings)
    generate_full_dataset(args.dataset_size // 4, 
                          folder=args.output_folder, 
                          split_name='val_indep',
                          independent=True,
                          num_processes=args.num_processes,
                          seed=args.seed,
                          settings=settings)
    generate_full_dataset(args.dataset_size // 10, 
                          folder=args.output_folder, 
                          split_name='val_triplets',
                          triplets=True,
                          start_data=os.path.join(args.output_folder, 'val.npz'),
                          num_processes=args.num_processes,
                          seed=args.seed,
                          settings=settings)
    generate_full_dataset(args.dataset_size // 10, 
                          folder=args.output_folder, 
                          split_name='test',
                          num_processes=args.num_processes,
                          seed=args.seed,
                          settings=settings)
    generate_full_dataset(args.dataset_size // 4, 
                          folder=args.output_folder, 
                          split_name='test_indep',
                          independent=True,
                          num_processes=args.num_processes,
                          seed=args.seed,
                          settings=settings)
    generate_full_dataset(args.dataset_size // 10, 
                          folder=args.output_folder, 
                          split_name='test_triplets',
                          triplets=True,
                          start_data=os.path.join(args.output_folder, 'test.npz'),
                          num_processes=args.num_processes,
                          seed=args.seed,
                          settings=settings)