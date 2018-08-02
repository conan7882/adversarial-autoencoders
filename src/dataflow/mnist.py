#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist.py
# Author: Qian Ge <geqian1001@gmail.com>

# from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorcv.dataflow.base import RNGDataFlow


DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

def identity(im):
    return im

def get_mnist_im_label(name, mnist_data):
    if name == 'train':
        return mnist_data.train.images, mnist_data.train.labels
    elif name == 'val':
        return mnist_data.validation.images, mnist_data.validation.labels
    else:
        return mnist_data.test.images, mnist_data.test.labels


class MNISTData(RNGDataFlow):
    def __init__(self, name, batch_dict_name=None, data_dir='', shuffle=True, pf=identity):
        assert os.path.isdir(data_dir)
        self._data_dir = data_dir

        self._shuffle = shuffle
        self._pf = pf

        if not isinstance(batch_dict_name, list):
            batch_dict_name = [batch_dict_name]
        self._batch_dict_name = batch_dict_name

        assert name in ['train', 'test', 'val']
        self.setup(epoch_val=0, batch_size=1)

        self._load_files(name)
        self._image_id = 0

    def next_batch_dict(self):
        batch_data = self.next_batch()
        data_dict = {key: data for key, data in zip(self._batch_dict_name, batch_data)}
        return data_dict

    def _load_files(self, name):
        mnist_data = input_data.read_data_sets(self._data_dir, one_hot=False)
        self.im_list = []
        self.label_list = []

        mnist_images, mnist_labels = get_mnist_im_label(name, mnist_data)

        for image, label in zip(mnist_images, mnist_labels):
            # TODO to be modified
            image = np.reshape(image, [28, 28, 1])
            image = self._pf(image)
            self.im_list.append(image)
            self.label_list.append(label)
        self.im_list = np.array(self.im_list)
        self.label_list = np.array(self.label_list)

        self._suffle_files()

    def _suffle_files(self):
        if self._shuffle:
            idxs = np.arange(self.size())

            self.rng.shuffle(idxs)
            self.im_list = self.im_list[idxs]
            self.label_list = self.label_list[idxs]

    def size(self):
        return self.im_list.shape[0]

    def next_batch(self):
        assert self._batch_size <= self.size(), \
          "batch_size {} cannot be larger than data size {}".\
           format(self._batch_size, self.size())
        start = self._image_id
        self._image_id += self._batch_size
        end = self._image_id
        batch_files = self.im_list[start:end]
        batch_label = self.label_list[start:end]

        if self._image_id + self._batch_size > self.size():
            self._epochs_completed += 1
            self._image_id = 0
            self._suffle_files()
        return [batch_files, batch_label]


