#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: aae.py
# Author: Qian Ge <geqian1001@gmail.com>

import tensorflow as tf
from src.models.base import BaseModel
import src.models.layers as L
import src.models.modules as modules
import src.models.ops as ops

# INIT_W = tf.keras.initializers.he_normal()
INIT_W = tf.contrib.layers.variance_scaling_initializer()


class AAE(BaseModel):
    def __init__(self, im_size=[28, 28], n_code=1000, n_channel=1, wd=0, label=False):
        self._n_channel = n_channel
        self._wd = wd
        self._n_code = n_code
        self._im_size = im_size
        self._flag_label = label
        self.layers = {}
        
    def create_generate_model(self, b_size):
        self.set_is_training(False)
        with tf.variable_scope('VAE', reuse=tf.AUTO_REUSE):
            self._create_generate_input()
            self.z = ops.tf_sample_standard_diag_guassian(b_size, self._n_code)
            self.layers['generate'] = tf.nn.sigmoid(self.decoder(self.z))

    def _create_generate_input(self):
        self.z = tf.placeholder(
            tf.float32, name='latent_z',
            shape=[None, self._n_code])
        self.keep_prob = 1.

    def create_train_model(self):
        self.set_is_training(True)
        self._create_train_input()
        with tf.variable_scope('VAE', reuse=tf.AUTO_REUSE):
            self.layers['encoder_out'] = self.encoder()
            self.layers['z'], self.layers['z_mu'], self.layers['z_std'], self.layers['z_log_std'] =\
                self.sample_latent()
            self.layers['decoder_out'] = self.decoder(self.layers['z'])

        self.layers['fake'] = self.discriminator(self.layers['z'])
        self.layers['real'] = self.discriminator(self.sample_prior())
        
    def _create_train_input(self):
        self.image = tf.placeholder(
            tf.float32, name='image',
            shape=[None, self._im_size[0], self._im_size[1], self._n_channel])
        self.label = tf.placeholder(
            tf.int32, name='label', shape=[None])
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def encoder(self):
        with tf.variable_scope('encoder'):
            # cnn_out = modules.encoder_CNN(
            #     self.image, is_training=self.is_training, init_w=INIT_W,
            #     wd=self._wd, bn=False, name='encoder_CNN')

            fc_out = modules.encoder_FC(self.image, self.is_training, keep_prob=self.keep_prob, wd=self._wd, name='encoder_FC', init_w=INIT_W)

            # fc_out = L.linear(
            #     out_dim=self._n_code*2, layer_dict=self.layers,
            #     inputs=cnn_out, init_w=INIT_W, wd=self._wd, name='Linear')

            return fc_out

    def sample_latent(self):
        with tf.variable_scope('sample_latent'):
            cnn_out = self.layers['encoder_out']
            
            z_mean = L.linear(
                out_dim=self._n_code, layer_dict=self.layers,
                inputs=cnn_out, init_w=INIT_W, wd=self._wd, name='latent_mean')
            z_std = L.linear(
                out_dim=self._n_code, layer_dict=self.layers, nl=L.softplus,
                inputs=cnn_out, init_w=INIT_W, wd=self._wd, name='latent_std')
            z_log_std = tf.log(z_std + 1e-8)

            b_size = tf.shape(cnn_out)[0]
            z = ops.tf_sample_diag_guassian(z_mean, z_std, b_size, self._n_code)
            return z, z_mean, z_std, z_log_std

    def decoder(self, inputs):
        with tf.variable_scope('decoder'):

            fc_out = modules.decoder_FC(inputs, self.is_training, keep_prob=self.keep_prob,
                                        wd=self._wd, name='decoder_FC', init_w=INIT_W)
            out_dim = self._im_size[0] * self._im_size[1] * self._n_channel
            decoder_out = L.linear(
                out_dim=out_dim, layer_dict=self.layers,
                inputs=fc_out, init_w=None, wd=self._wd, name='decoder_linear')
            decoder_out = tf.reshape(decoder_out, (-1, self._im_size[0], self._im_size[1], self._n_channel))

            return decoder_out

    def discriminator(self, inputs):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            fc_out = modules.discriminator_FC(inputs, self.is_training,
                                              wd=self._wd, name='discriminator_FC',
                                              init_w=INIT_W)
            return fc_out

    def sample_prior(self):
        b_size = tf.shape(self.image)[0]
        return ops.tf_sample_standard_diag_guassian(b_size, self._n_code)

    def _get_loss(self):
        with tf.name_scope('loss'):
            with tf.name_scope('likelihood'):
                p_hat = tf.nn.sigmoid(self.layers['decoder_out'], name='estimate_prob')
                p = self.image
                cross_entropy = p * tf.log(p_hat + 1e-6) + (1 - p) * tf.log(1 - p_hat + 1e-6)
                cross_entropy_loss = -tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=[1,2,3]))

            with tf.name_scope('GAN'):
                label = tf.ones_like(self.layers['fake'])
                fake_logits = self.layers['fake']
                gan_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=label,
                    logits=fake_logits,
                    name='gan_loss')
                self.gan_loss = tf.reduce_mean(gan_loss)

            return cross_entropy_loss + self.gan_loss

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(self.lr)

    def get_train_op(self):
        with tf.name_scope('train'):
            opt = self.get_optimizer()
            loss = self.get_loss()
            var_list = tf.trainable_variables(scope='VAE')
            grads = tf.gradients(loss, var_list)
            [tf.summary.histogram('gradient/' + var.name, grad, 
             collections=['train']) for grad, var in zip(grads, var_list)]
            return opt.apply_gradients(zip(grads, var_list))

    def get_discrimator_train_op(self):
        with tf.name_scope('discrimator_train_op'):
            with tf.name_scope('discrimator_loss'):
                loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(self.layers['real']),
                    logits=self.layers['real'],
                    name='loss_real')
                loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(self.layers['fake']),
                    logits=self.layers['fake'],
                    name='loss_fake')
                d_loss = loss_real + loss_fake
                self.d_loss = tf.reduce_mean(d_loss)
                
            opt = tf.train.AdamOptimizer(self.lr)
            var_list = tf.trainable_variables(scope='discriminator')
            # print(tf.trainable_variables())
            grads = tf.gradients(self.d_loss, var_list)
            # [tf.summary.histogram('gradient/' + var.name, grad, 
            #  collections=['train']) for grad, var in zip(grads, var_list)]
            return opt.apply_gradients(zip(grads, var_list))

    def get_valid_summary(self):
        with tf.name_scope('generate'):
            tf.summary.image(
                'image',
                tf.cast(self.layers['generate'], tf.float32),
                collections=['generate'])
        return tf.summary.merge_all(key='generate')

    def get_train_summary(self):
        tf.summary.image(
            'input_image',
            tf.cast(self.image, tf.float32),
            collections=['train'])
        tf.summary.image(
            'out_image',
            tf.cast(tf.nn.sigmoid(self.layers['decoder_out']), tf.float32),
            collections=['train'])
        return tf.summary.merge_all(key='train')

