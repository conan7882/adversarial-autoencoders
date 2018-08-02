#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: trainer.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
# import scipy.misc
import numpy as np
import tensorflow as tf

import src.utils.viz as viz
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

def display(global_step,
            step,
            scaler_sum_list,
            name_list,
            collection,
            summary_val=None,
            summary_writer=None,
            ):
    print('[step: {}]'.format(global_step), end='')
    for val, name in zip(scaler_sum_list, name_list):
        print(' {}: {:.4f}'.format(name, val * 1. / step), end='')
    print('')
    if summary_writer is not None:
        s = tf.Summary()
        for val, name in zip(scaler_sum_list, name_list):
            s.value.add(tag='{}/{}'.format(collection, name),
                        simple_value=val * 1. / step)
        summary_writer.add_summary(s, global_step)
        if summary_val is not None:
            summary_writer.add_summary(summary_val, global_step)

class Trainer(object):
    def __init__(self, train_model, generate_model, train_data, init_lr=1e-3, save_path=None):

        self._save_path = save_path

        self._t_model = train_model
        self._g_model = generate_model
        self._train_data = train_data
        self._lr = init_lr

        self._train_op = train_model.get_train_op()
        self._loss_op = train_model.get_loss()
        self._train_summary_op = train_model.get_train_summary()

        self._generate_op = generate_model.layers['generate']
        self._valid_summary_op = generate_model.get_valid_summary()

        self.global_step = 0

    def train_epoch(self, sess, summary_writer=None):
        self._t_model.set_is_training(True)
        display_name_list = ['loss']
        cur_summary = None

        cur_epoch = self._train_data.epochs_completed

        step = 0
        loss_sum = 0
        while cur_epoch == self._train_data.epochs_completed:
            self.global_step += 1
            step += 1

            batch_data = self._train_data.next_batch_dict()
            im = batch_data['im']
            label = batch_data['label']
            _, loss, cur_summary = sess.run(
                [self._train_op, self._loss_op, self._train_summary_op], 
                feed_dict={self._t_model.image: im,
                           self._t_model.lr: self._lr,
                           self._t_model.keep_prob: 0.9})

            loss_sum += loss

            if step % 100 == 0:
                display(self.global_step,
                    step,
                    [loss_sum],
                    display_name_list,
                    'train',
                    summary_val=cur_summary,
                    summary_writer=summary_writer)

        print('==== epoch: {}, lr:{} ===='.format(cur_epoch, self._lr))
        display(self.global_step,
                step,
                [loss_sum],
                display_name_list,
                'train',
                summary_val=cur_summary,
                summary_writer=summary_writer)

    def valid_epoch(self, sess, batch_size=128, summary_writer=None):
        # self._g_model.set_is_training(True)
        # display_name_list = ['loss']
        # cur_summary = None
        # dataflow.setup(epoch_val=0, batch_size=batch_size)

        step = 0
        # while dataflow.epochs_completed == 0:
        for i in range(10):
            self.global_step += 1
            step += 1
            cur_summary, gen_im = sess.run([self._valid_summary_op, self._generate_op])
            break

        if self._save_path:
            im_save_path = os.path.join(self._save_path,
                                        'generate_step_{}.png'.format(self.global_step))
            viz.viz_filters(batch_im=gen_im, grid_size=[10, 10],
                            save_path=im_save_path, gap=0, gap_color=0,
                            shuffle=False)
        if summary_writer:
            cur_summary = sess.run(self._valid_summary_op)
            summary_writer.add_summary(cur_summary, self.global_step)


        # print('==== epoch: {}, lr:{} ===='.format(cur_epoch, self._lr))
        # display(self.global_step,
        #         step,
        #         [loss_sum],
        #         display_name_list,
        #         'train',
        #         summary_val=cur_summary,
        #         summary_writer=summary_writer)


    # def valid_epoch(self, sess, dataflow, batch_size):
    #     # self._model.set_is_training(False)
    #     dataflow.setup(epoch_val=0, batch_size=batch_size)

    #     step = 0
    #     loss_sum = 0
    #     acc_sum = 0
    #     while dataflow.epochs_completed == 0:
    #         step += 1
    #         batch_data = dataflow.next_batch_dict()
    #         loss, acc = sess.run(
    #             [self._loss_op, self._accuracy_op], 
    #             feed_dict={self._model.image: batch_data['im'],
    #                        self._model.label: batch_data['label'],
    #                        })
    #         loss_sum += loss
    #         acc_sum += acc
    #     print('valid loss: {:.4f}, accuracy: {:.4f}'
    #           .format(loss_sum * 1.0 / step, acc_sum * 1.0 / step))

    #     # self._model.set_is_training(True)

    # def test_batch(self, sess, batch_data, unit_pixel, size, scale, save_path=''):
    #     def draw_bbx(ax, x, y):
    #         rect = patches.Rectangle(
    #             (x, y), cur_size, cur_size, edgecolor='r', facecolor='none', linewidth=2)
    #         ax.add_patch(rect)

    #     self._model.set_is_training(False)
        
    #     test_im = batch_data['im']
    #     loc_list, pred, input_im, glimpses = sess.run(
    #         [self._sample_loc_op, self._pred_op, self._model.input_im,
    #          self._model.layers['retina_reprsent']],
    #         feed_dict={self._model.image: test_im,
    #                    self._model.label: batch_data['label'],
    #                     })

    #     pad_r = size * (2 ** (scale - 2))
    #     print(pad_r)
    #     im_size = input_im[0].shape[0]
    #     loc_list = np.clip(np.array(loc_list), -1.0, 1.0)
    #     loc_list = loc_list * 1.0 * unit_pixel / (im_size / 2 + pad_r)
    #     loc_list = (loc_list + 1.0) * 1.0 / 2 * (im_size + pad_r * 2)
    #     offset = pad_r

    #     print(pred)
    #     for step_id, cur_loc in enumerate(loc_list):
    #         im_id = 0
    #         glimpse = glimpses[step_id]
    #         for im, loc, cur_glimpse in zip(input_im, cur_loc, glimpse):
    #             im_id += 1                
    #             fig, ax = plt.subplots(1)
    #             ax.imshow(np.squeeze(im), cmap='gray')
    #             for scale_id in range(0, scale):
    #                 cur_size = size * 2 ** scale_id
    #                 side = cur_size * 1.0 / 2
    #                 x = loc[1] - side - offset
    #                 y = loc[0] - side - offset
    #                 draw_bbx(ax, x, y)
    #             # plt.show()
    #             for i in range(0, scale):
    #                 scipy.misc.imsave(
    #                     os.path.join(save_path,'im_{}_glimpse_{}_step_{}.png').format(im_id, i, step_id),
    #                     np.squeeze(cur_glimpse[:,:,i]))
    #             plt.savefig(os.path.join(
    #                 save_path,'im_{}_step_{}.png').format(im_id, step_id))
    #             plt.close(fig)

    #     self._model.set_is_training(True)
