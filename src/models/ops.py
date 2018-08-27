#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ops.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import tensorflow_probability as tfp


def tf_sample_standard_diag_guassian(b_size, n_code):
    mean_list = [0.0 for i in range(0, n_code)]
    std_list = [1.0 for i in range(0, n_code)]
    mvn = tfp.distributions.MultivariateNormalDiag(
        loc=mean_list,
        scale_diag=std_list)
    samples = mvn.sample(sample_shape=(b_size,), seed=None, name='sample')
    return samples

def tf_sample_diag_guassian(mean, std, b_size, n_code):
    mean_list = [0.0 for i in range(0, n_code)]
    std_list = [1.0 for i in range(0, n_code)]
    mvn = tfp.distributions.MultivariateNormalDiag(
        loc=mean_list,
        scale_diag=std_list)
    samples = mvn.sample(sample_shape=(b_size,), seed=None, name='sample')
    samples = mean +  tf.multiply(std, samples)

    return samples