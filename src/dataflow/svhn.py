#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: svhn.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
from scipy.io import loadmat
from src.dataflow.mnist import MNISTData

def identity(im):
    return im

class CenterSVHN(MNISTData):
    def _load_files(self, name):
        if name == 'train':
            name = 'train_32x32.mat'
        else:
            name = 'test_32x32.mat'

        data_mat = loadmat(file_name)
        label_list = data_mat['y'].astype(np.int32)
        im_list = data_mat['X'].astype(np.float32) 


        # def read_h_im(file_name):
        #     mat = loadmat(file_name)
        #     h_im = mat['h_im'].astype(np.float32)
        #     return h_im


        # mnist_data = input_data.read_data_sets(self._data_dir, one_hot=False)
        # self.im_list = []
        # self.label_list = []

        # mnist_images, mnist_labels = get_mnist_im_label(name, mnist_data)

        # for image, label in zip(mnist_images, mnist_labels):
        #     # TODO to be modified
        #     image = np.reshape(image, [28, 28, 1])
        #     image = self._pf(image)
        #     self.im_list.append(image)
        #     self.label_list.append(label)
        # self.im_list = np.array(self.im_list)
        # self.label_list = np.array(self.label_list)

        # self._suffle_files()


