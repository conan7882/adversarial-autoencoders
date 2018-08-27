#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: viz.py
# Author: Qian Ge <geqian1001@gmail.com>

import numpy as np
import imageio


def viz_batch_im(batch_im, grid_size, save_path,
                gap=0, gap_color=0, shuffle=False):

    batch_im = np.array(batch_im)
    if len(batch_im.shape) == 4:
        n_channel = batch_im.shape[-1]
    elif len(batch_im.shape) == 3:
        n_channel = 1
        batch_im = np.expand_dims(batch_im, axis=-1)
    assert len(grid_size) == 2

    h = batch_im.shape[1]
    w = batch_im.shape[2]

    merge_im = np.zeros((h * grid_size[0] + (grid_size[0] + 1) * gap,
                         w * grid_size[1] + (grid_size[1] + 1) * gap,
                         n_channel)) + gap_color

    n_viz_filter = min(batch_im.shape[0], grid_size[0] * grid_size[1])
    if shuffle == True:
        pick_id = np.random.permutation(batch_im.shape[0])
    else:
        pick_id = range(0, batch_im.shape[0])
    for idx in range(0, n_viz_filter):
        i = idx % grid_size[1]
        j = idx // grid_size[1]
        cur_filter = batch_im[pick_id[idx], :, :, :]
        merge_im[j * (h + gap) + gap: j * (h + gap) + h + gap,
                 i * (w + gap) + gap: i * (w + gap) + w + gap, :]\
            = (cur_filter)
    imageio.imwrite(save_path, np.squeeze(merge_im))


