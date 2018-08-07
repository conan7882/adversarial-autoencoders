#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: generator.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import math
import numpy as np
import tensorflow as tf

import src.utils.viz as viz
import src.models.distribution as distribution


class Generator(object):
    def __init__(self, generate_model, distr_type='gaussian',
                 n_labels=None, use_label=False, save_path=None):

        self._save_path = save_path
        self._g_model = generate_model
        self._generate_op = generate_model.layers['generate']

        self._dist = distr_type
        self._n_labels = n_labels
        self._use_label = use_label
        if self._use_label:
            assert self._n_labels is not None

    def generate_samples(self, sess, plot_size, manifold=False, file_id=None):
        # if z is None:
        #     gen_im = sess.run(self._generate_op)
        # else:
        n_samples = plot_size * plot_size

        label_indices = None
        if self._use_label:
            cur_r = 0
            label_indices = []
            cur_label = -1
            while cur_r < plot_size:
                cur_label = cur_label + 1 if cur_label < self._n_labels - 1 else 0
                row_label = np.ones(plot_size) * cur_label
                label_indices.extend(row_label)
                cur_r += 1

        if manifold:
            if self._dist ==  'gaussian':
                random_code = distribution.interpolate(
                    plot_size=plot_size, interpolate_range=[-3, 3, -3, 3])
                self.viz_samples(sess, random_code, plot_size, file_id=file_id)
            else:
                for mode_id in range(self._n_labels):
                    random_code = distribution.interpolate_gm(
                        plot_size=plot_size, interpolate_range=[-1., 1., -0.2, 0.2],
                        mode_id=mode_id, n_mode=self._n_labels)
                    self.viz_samples(sess, random_code, plot_size,
                                     file_id='{}_{}'.format(file_id, mode_id))
        else:
            if self._dist ==  'gaussian':
                random_code = distribution.diagonal_gaussian(
                    n_samples, self._g_model.n_code, mean=0, var=1.0)
            else:
                random_code = distribution.gaussian_mixture(
                    n_samples, n_dim=self._g_model.n_code, n_labels=self._n_labels,
                    x_var=0.5, y_var=0.1, label_indices=label_indices)

            self.viz_samples(sess, random_code, plot_size, file_id=file_id)

    def viz_samples(self, sess, random_code, plot_size, file_id=None):
        gen_im = sess.run(self._generate_op, feed_dict={self._g_model.z: random_code})
        if self._save_path:
            if file_id is not None:
                im_save_path = os.path.join(
                    self._save_path, 'generate_im_{}.png'.format(file_id))
            else:
                im_save_path = os.path.join(
                    self._save_path, 'generate_im.png')

            n_sample = len(gen_im)
            plot_size = int(min(plot_size, math.sqrt(n_sample)))
            viz.viz_batch_im(batch_im=gen_im, grid_size=[plot_size, plot_size],
                            save_path=im_save_path, gap=0, gap_color=0,
                            shuffle=False)



            

