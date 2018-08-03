#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: distribution.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np


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
