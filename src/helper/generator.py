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

    def random_generate(self, sess, batch_size=128, z=None):
        if z is None:
            gen_im = sess.run(self._generate_op)
        else:
            gen_im = sess.run(self._generate_op, feed_dict={self._g_model.z: z})
        if self._save_path:
            im_save_path = os.path.join(self._save_path,
                                        'random_generate.png')
            viz.viz_filters(batch_im=gen_im, grid_size=[20, 20],
                            save_path=im_save_path, gap=0, gap_color=0,
                            shuffle=False)
