#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
from abc import abstractmethod


class BaseModel(object):
    """ Model with single loss and single optimizer """

    def set_is_training(self, is_training=True):
        self.is_training = is_training

    def get_loss(self):
        try:
            return self._loss
        except AttributeError:
            self._loss = self._get_loss()
        return self._loss

    def _get_loss(self):
        raise NotImplementedError()

    def get_optimizer(self):
        try:
            return self.optimizer
        except AttributeError:
            self.optimizer = self._get_optimizer()
        return self.optimizer

    def _get_optimizer(self):
        raise NotImplementedError()

    def get_train_op(self):
        with tf.name_scope('train'):
            opt = self.get_optimizer()
            loss = self.get_loss()
            var_list = tf.trainable_variables()
            grads = tf.gradients(loss, var_list)
            # [tf.summary.histogram('gradient/' + var.name, grad, 
            #  collections=['train']) for grad, var in zip(grads, var_list)]
            return opt.apply_gradients(zip(grads, var_list))


