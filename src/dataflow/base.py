#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import numpy as np 
import lib.utils.utils as utils
# from tensorcv.dataflow.common import get_file_list
from tensorcv.dataflow.base import RNGDataFlow

def get_file_list(file_dir, file_ext, sub_name=None):
    # assert file_ext in ['.mat', '.png', '.jpg', '.jpeg']
    re_list = []

    if sub_name is None:
        return np.array([os.path.join(root, name)
            for root, dirs, files in os.walk(file_dir) 
            for name in sorted(files) if name.endswith(file_ext)])
    else:
        return np.array([os.path.join(root, name)
            for root, dirs, files in os.walk(file_dir) 
            for name in sorted(files) if name.endswith(file_ext) and sub_name in name])

class DataFlow(RNGDataFlow):
    def __init__(self,
                 data_name_list,
                 # data_type_list,
                 # n_channel_list,
                 data_dir='',
                 shuffle=True,
                 batch_dict_name=None,
                 load_fnc_list=None,
                 # pf_list=None,
                 ):
        data_name_list = utils.make_list(data_name_list)
        # data_type_list = utils.make_list(data_type_list)
        # n_channel_list = utils.make_list(n_channel_list)
        load_fnc_list = utils.make_list(load_fnc_list)
        # pf_list = utils.make_list(pf_list)
        utils.assert_len([data_name_list, load_fnc_list])
        self._n_dataflow = len(data_name_list)
        # self._data_name_list = data_name_list
        # self._data_type_list = data_type_list
        # self._n_channel_list = n_channel_list
        # pf_list = [pf if pf is not None else identity for pf in pf_list]
        # self._pf_list = pf_list
        self._load_fnc_list = load_fnc_list

        self._data_dir = data_dir
        self._shuffle = shuffle
        self._batch_dict_name = batch_dict_name

        self._data_id = 0
        self.setup(epoch_val=0, batch_size=1)
        self._load_file_list(data_name_list)

    def size(self):
        return len(self._file_name_list[0])

    def _load_file_list(self, data_name_list):
        data_dir = self._data_dir
        self._file_name_list = []
        for data_name in data_name_list:
            self._file_name_list.append(get_file_list(data_dir, data_name))
        if self._shuffle:
            self._suffle_file_list()

    def _suffle_file_list(self):
        idxs = np.arange(self.size())
        self.rng.shuffle(idxs)
        for idx, file_list in enumerate(self._file_name_list):
            self._file_name_list[idx] = file_list[idxs]

    def next_batch(self):
        assert self._batch_size <= self.size(), \
        "batch_size cannot be larger than data size"

        if self._data_id + self._batch_size > self.size():
            start = self._data_id
            end = self.size()
        else:
            start = self._data_id
            self._data_id += self._batch_size
            end = self._data_id
        batch_data = self._load_data(start, end)

        if end == self.size():
            self._epochs_completed += 1
            self._data_id = 0
            if self._shuffle:
                self._suffle_file_list()
        return batch_data

    def _load_data(self, start, end):
        data_list = [[] for i in range(0, self._n_dataflow)]
        for k in range(start, end):
            for read_idx, read_fnc in enumerate(self._load_fnc_list):
                data = read_fnc(self._file_name_list[read_idx][k])
                data_list[read_idx].append(data)

        for idx, data in enumerate(data_list):
            data_list[idx] = np.array(data)

        return data_list

    def next_batch_dict(self):
        batch_data = self.next_batch()
        return {key: data for key, data in zip(self._batch_dict_name, batch_data)} 
