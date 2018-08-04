#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: distribution.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
from math import sin,cos,sqrt


def interpolate(plot_size=20):
    """ Util to interpolate between two points in n-dimensional latent space
        Borrow from:
        https://github.com/fastforwardlabs/vae-tf/blob/master/plot.py#L85
    """
    # zs = np.array([np.linspace(start, end, n_code) # interpolate across every z dimension
    #                for start, end in zip(latent_1, latent_2)]).T
    nx=plot_size
    ny=plot_size
    range_=(-4, 4)
    min_, max_ = range_
    
    zs = np.rollaxis(np.mgrid[max_:min_:ny*1j, min_:max_:nx*1j], 0, 3)
    return zs

def gaussian(batch_size, n_dim, mean=0, var=1):
    z = np.random.normal(mean, var, (batch_size, n_dim)).astype(np.float32)
    return z

def gaussian_mixture(batch_size, n_dim=2, n_labels=10, x_var=0.5, y_var=0.1, label_indices=None):
    if n_dim % 2 != 0:
        raise Exception("n_dim must be a multiple of 2.")

    def sample(x, y, label, n_labels):
        shift = 1.4
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
