#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: visualizer.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# import src.utils.viz as viz

class Visualizer(object):
    def __init__(self, model, save_path=None):

        self._save_path = save_path
        self._model = model
        self._latent_op = model.layers['z']

    def viz_2Dlatent_variable(self, sess, dataflow, batch_size=128):
        """
        modify from:
        https://github.com/fastforwardlabs/vae-tf/blob/master/plot.py#L45
        """
        dataflow.setup(epoch_val=0, batch_size=batch_size)
        latent_var_list = []
        label_list = []
        while dataflow.epochs_completed == 0:
            batch_data = dataflow.next_batch_dict()
            im = batch_data['im']
            labels = batch_data['label']
            latent_var = sess.run(
                self._latent_op, 
                feed_dict={self._model.image: im,
                           self._model.keep_prob: 1.})
            latent_var_list.extend(latent_var)
            label_list.extend(labels)
            # print(latent_var)

        ys, xs = np.array(latent_var_list).T

        plt.figure()
        # plt.title("round {}: {} in latent space".format(model.step, title))
        kwargs = {'alpha': 0.8}

        classes = set(label_list)
        if classes:
            colormap = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
            kwargs['c'] = [colormap[i] for i in label_list]

            # make room for legend
            ax = plt.subplot(111)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            handles = [mpatches.Circle((0,0), label=class_, color=colormap[i])
                        for i, class_ in enumerate(classes)]
            ax.legend(handles=handles, shadow=True, bbox_to_anchor=(1.05, 0.45),
                        fancybox=True, loc='center left')

        plt.scatter(xs, ys, **kwargs)

        # if range_:
        #     plt.xlim(*range_)
        #     plt.ylim(*range_)
        fig_save_path = os.path.join(self._save_path, 'latent.png')
        plt.savefig(fig_save_path, bbox_inches="tight")
        # plt.show()
        # if save:
        #     title = "{}_latent_{}_round_{}_{}.png".format(
        #         model.datetime, "_".join(map(str, model.architecture)),
        #         model.step, name)
        #     plt.savefig(os.path.join(outdir, title), bbox_inches="tight")