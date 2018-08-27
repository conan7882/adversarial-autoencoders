#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: distribution.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
from math import sin,cos,sqrt


def interpolate(plot_size=20, interpolate_range=[-3, 3, -3, 3]):
    assert len(interpolate_range) == 4
    nx = plot_size
    ny = plot_size
    min_x = interpolate_range[0]
    max_x = interpolate_range[1]
    min_y = interpolate_range[2]
    max_y = interpolate_range[3]
    
    zs = np.rollaxis(np.mgrid[min_x: max_x: nx*1j, max_y:min_y: ny*1j], 0, 3)
    zs = zs.transpose(1, 0, 2)
    return np.reshape(zs, (plot_size*plot_size, 2))

def interpolate_gm(plot_size=20, interpolate_range=[-1., 1., -0.2, 0.2],
                   mode_id=0, n_mode=10):
    n_samples = plot_size * plot_size
    def sample(x, y, mode_id, n_mode):
        shift = 1.4
        r = 2.0 * np.pi / float(n_mode) * float(mode_id)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    interp_grid = interpolate(plot_size=20, interpolate_range=interpolate_range)
    x = interp_grid[:, 0]
    y = interp_grid[:, 1]

    z = np.empty((n_samples, 2), dtype=np.float32)
    for i in range(n_samples):
        z[i, :2] = sample(x[i], y[i], mode_id, n_mode)
    return z

def gaussian(batch_size, n_dim, mean=0, var=1.):
    z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)
    return z

def diagonal_gaussian(batch_size, n_dim, mean=0, var=1.):
    cov_mat = np.diag([var for i in range(n_dim)])
    mean_vec = [mean for i in range(n_dim)]
    z = np.random.multivariate_normal(
        mean_vec, cov_mat, (batch_size,)).astype(np.float32)
    return z

def gaussian_mixture(batch_size, n_dim=2, n_labels=10,
                     x_var=0.5, y_var=0.1, label_indices=None):
    # borrow from:
    # https://github.com/nicklhy/AdversarialAutoEncoder/blob/master/data_factory.py#L40
    if n_dim % 2 != 0:
        raise Exception("n_dim must be a multiple of 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
        if label >= n_labels:
            label =  np.random.randint(0, n_labels)
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0, x_var, (batch_size, n_dim // 2))
    y = np.random.normal(0, y_var, (batch_size, n_dim // 2))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range(n_dim // 2):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels), n_labels)

    return z
