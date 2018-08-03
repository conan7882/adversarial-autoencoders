#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: generator.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np
import tensorflow as tf

import src.utils.viz as viz

class Generator(object):
    def __init__(self, generate_model, save_path=None):

        self._save_path = save_path

        self._g_model = generate_model

        self._generate_op = generate_model.layers['generate']
        # self._valid_summary_op = generate_model.get_valid_summary()

    def generate_samples(self, sess, plot_size, z=None, file_id=None):
        if z is None:
            gen_im = sess.run(self._generate_op)
        else:
            gen_im = sess.run(self._generate_op, feed_dict={self._g_model.z: z})
        if self._save_path:
            if file_id is not None:
                im_save_path = os.path.join(
                    self._save_path, 'generate_im_{}.png'.format(file_id))
            else:
                im_save_path = os.path.join(
                    self._save_path, 'generate_im.png')
            viz.viz_batch_im(batch_im=gen_im, grid_size=[plot_size, plot_size],
                            save_path=im_save_path, gap=0, gap_color=0,
                            shuffle=False)


            

