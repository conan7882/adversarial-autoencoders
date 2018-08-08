#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: modules.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
import src.models.layers as L


def encoder_FC(inputs, is_training, n_hidden=1000, nl=tf.nn.relu,
               keep_prob=0.5, wd=0, name='encoder_FC', init_w=None):
    layer_dict = {}
    layer_dict['cur_input'] = inputs
    with tf.variable_scope(name):
        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.linear],
                       out_dim=n_hidden, layer_dict=layer_dict, init_w=init_w,
                       wd=wd):
            L.linear(name='linear1', nl=nl)
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)
            L.linear(name='linear2', nl=nl)
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)

        return layer_dict['cur_input']
        
def decoder_FC(inputs, is_training, n_hidden=1000, nl=tf.nn.relu,
               keep_prob=0.5, wd=0, name='decoder_FC', init_w=None):
    layer_dict = {}
    layer_dict['cur_input'] = inputs
    with tf.variable_scope(name):
        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.linear],
                       out_dim=n_hidden, layer_dict=layer_dict, init_w=init_w,
                       wd=wd):
            L.linear(name='linear1', nl=nl)
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)
            L.linear(name='linear2', nl=nl)
            L.drop_out(layer_dict, is_training, keep_prob=keep_prob)

        return layer_dict['cur_input']

def discriminator_FC(inputs, is_training, n_hidden=1000, nl=tf.nn.relu,
                     wd=0, name='discriminator_FC', init_w=None):
    layer_dict = {}
    layer_dict['cur_input'] = inputs
    with tf.variable_scope(name):
        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.linear],
                       layer_dict=layer_dict, init_w=init_w,
                       wd=wd):
            L.linear(name='linear1', nl=nl, out_dim=n_hidden)
            L.linear(name='linear2', nl=nl, out_dim=n_hidden)
            L.linear(name='output', out_dim=1)

        return layer_dict['cur_input']

def encoder_CNN(inputs, is_training, wd=0, bn=False, name='encoder_CNN',
                init_w=tf.keras.initializers.he_normal()):
    # init_w = tf.keras.initializers.he_normal()
    layer_dict = {}
    layer_dict['cur_input'] = inputs
    with tf.variable_scope(name):
        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.conv],
                        layer_dict=layer_dict, bn=bn, nl=tf.nn.relu,
                        init_w=init_w, padding='SAME', pad_type='ZERO',
                        is_training=is_training, wd=0):
            
            L.conv(filter_size=5, out_dim=32, name='conv1', add_summary=False)
            L.max_pool(layer_dict, name='pool1')
            L.conv(filter_size=3, out_dim=64, name='conv2', add_summary=False)
            L.max_pool(layer_dict, name='pool2')
            # L.conv(filter_size=3, out_dim=128, name='conv3', add_summary=False)
            # L.max_pool(layer_dict, name='pool3')

            return layer_dict['cur_input']

def decoder_CNN(inputs, is_training, out_channel=1, wd=0, bn=False, name='decoder_CNN',
                init_w=tf.keras.initializers.he_normal()):
    # init_w = tf.keras.initializers.he_normal()
    layer_dict = {}
    layer_dict['cur_input'] = inputs
    with tf.variable_scope(name):
        arg_scope = tf.contrib.framework.arg_scope
        with arg_scope([L.transpose_conv],
                        layer_dict=layer_dict, nl=tf.nn.relu, stride=2,
                        init_w=init_w, wd=0):
            
            # L.transpose_conv(filter_size=3, out_dim=64, name='deconv1')
            L.transpose_conv(filter_size=3, out_dim=32, name='deconv2')
            L.transpose_conv(filter_size=3, out_dim=out_channel, name='deconv3')

            return layer_dict['cur_input']

def train_discrimator(fake_in, real_in, loss_weight, opt, var_list, name):
    with tf.name_scope(name):
        with tf.name_scope('discrimator_loss'):
            loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(real_in),
                logits=real_in,
                name='loss_real')
            loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake_in),
                logits=fake_in,
                name='loss_fake')
            d_loss = tf.reduce_mean(loss_real) + tf.reduce_mean(loss_fake)
                
            # opt = tf.train.AdamOptimizer(lr, beta1=0.5)
            # opt = tf.train.MomentumOptimizer(self.lr, momentum=0.1)
            # dc_var = [var for var in all_variables if 'dc_' in var.name]
            # var_list = tf.trainable_variables(scope='discriminator')
            # print(tf.trainable_variables())
            # print(var_list)
            grads = tf.gradients(d_loss * loss_weight, var_list)
            # [tf.summary.histogram('gradient/' + var.name, grad, 
            #  collections=['train']) for grad, var in zip(grads, var_list)]
        train_op = opt.apply_gradients(zip(grads, var_list))

        return d_loss, train_op

def train_generator(fake_in, loss_weight, opt, var_list, name):
     with tf.name_scope(name):
        with tf.name_scope('generator_loss'):
            loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(fake_in),
                logits=fake_in,
                name='loss_fake')
            g_loss = tf.reduce_mean(loss_fake)
        # opt = tf.train.AdamOptimizer(lr, beta1=0.5)
        # print(var_list)
        grads = tf.gradients(g_loss * loss_weight, var_list)
        train_op = opt.apply_gradients(zip(grads, var_list))

        return g_loss, train_op

def train_by_cross_entropy_loss(logits, labels, loss_weight, opt, var_list, name):
    with tf.name_scope(name):
        with tf.name_scope('cross_entropy_loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels,
                logits=logits,
                name='cross_entropy')
            cross_entropy = tf.reduce_mean(cross_entropy)
        # opt = tf.train.AdamOptimizer(lr, beta1=0.5)
        # print(var_list)
        grads = tf.gradients(cross_entropy * loss_weight, var_list)
        train_op = opt.apply_gradients(zip(grads, var_list))

        return cross_entropy, train_op

