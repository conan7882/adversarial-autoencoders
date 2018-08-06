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
# INIT_W = tf.random_normal_initializer(mean=0., stddev=0.01)

class AAE(BaseModel):
    def __init__(self, im_size=[28, 28], n_code=1000, n_channel=1, wd=0,
                 use_label=False, n_class=None, use_supervise=False,
                 enc_weight=1., gen_weight=1., dis_weight=1.):
        self._n_channel = n_channel
        self._wd = wd
        self.n_code = n_code
        self._im_size = im_size
        if use_supervise:
            use_label = False
        self._flag_label = use_label
        self._flag_supervise = use_supervise
        self._n_class = n_class
        self._enc_w = enc_weight
        self._gen_w = gen_weight
        self._dis_w = dis_weight
        self.layers = {}
        
    def create_generate_model(self, b_size):
        self.set_is_training(False)
        with tf.variable_scope('AE', reuse=tf.AUTO_REUSE):
            self._create_generate_input()
            self.z = ops.tf_sample_standard_diag_guassian(b_size, self.n_code)
            decoder_in = self.z
            if self._flag_supervise:
                label = []
                for i in range(self._n_class):
                    label.extend([i for k in range(10)])
                # label = [i for i in range(self._n_class)]
                self.label = tf.convert_to_tensor(label) # [n_class]
                one_hot_label = tf.one_hot(self.label, self._n_class) # [n_class*10, n_class]
                # one_hot_label = tf.tile(one_hot_label, [10, 1]) # [n_class*10, n_class]
                # one_hot_label = tf.transpose()
                choose_code = decoder_in[:self._n_class] # [n_class, n_code]
                choose_code = tf.tile(choose_code, [10, 1]) # [n_class*10, n_code]
                decoder_in = tf.concat((choose_code, one_hot_label), axis=-1)
            self.layers['generate'] = (self.decoder(decoder_in) + 1. ) / 2.
            # self.layers['generate'] = tf.nn.sigmoid(self.decoder(self.z))

    def _create_generate_input(self):
        self.z = tf.placeholder(
            tf.float32, name='latent_z',
            shape=[None, self.n_code])
        self.keep_prob = 1.

    def create_train_model(self):
        self.set_is_training(True)
        self._create_train_input()
        with tf.variable_scope('AE', reuse=tf.AUTO_REUSE):
            encoder_in = self.image
            if self.is_training:
                encoder_in += tf.random_normal(
                    tf.shape(encoder_in),
                    mean=0.0,
                    stddev=0.6,
                    dtype=tf.float32)
            self.encoder_in = encoder_in
            self.layers['encoder_out'] = self.encoder(self.encoder_in)
            self.layers['z'], self.layers['z_mu'], self.layers['z_std'], self.layers['z_log_std'] =\
                self.sample_latent()

            decoder_in = self.layers['z']
            if self._flag_supervise:
                one_hot_label = tf.one_hot(self.label, self._n_class)
                decoder_in = tf.concat((decoder_in, one_hot_label), axis=-1)
            self.layers['decoder_out'] = self.decoder(decoder_in)
            self.layers['sample_im'] = (self.layers['decoder_out'] + 1. ) / 2.

        fake_in = self.layers['z']
        real_in = self.real_distribution
        if self._flag_label:
            # convert labels to one-hot vectors, add one digit for data without label
            one_hot_label = tf.one_hot(self.label, self._n_class + 1)
            fake_in = tf.concat((fake_in, one_hot_label), axis=-1)
            real_in = tf.concat((real_in, one_hot_label), axis=-1)
        self.layers['fake'] = self.discriminator(fake_in)
        # self.layers['real_distribution'] = self.sample_prior()
        self.layers['real'] = self.discriminator(real_in)
        
    def _create_train_input(self):
        self.image = tf.placeholder(
            tf.float32, name='image',
            shape=[None, self._im_size[0], self._im_size[1], self._n_channel])
        self.label = tf.placeholder(
            tf.int32, name='label', shape=[None])
        self.real_distribution = tf.placeholder(
            tf.float32, name='real_distribution', shape=[None, self.n_code])
        self.lr = tf.placeholder(tf.float32, name='lr')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def encoder(self, inputs):
        with tf.variable_scope('encoder'):
            # cnn_out = modules.encoder_CNN(
            #     self.image, is_training=self.is_training, init_w=INIT_W,
            #     wd=self._wd, bn=False, name='encoder_CNN')

            fc_out = modules.encoder_FC(inputs, self.is_training, keep_prob=self.keep_prob, wd=self._wd, name='encoder_FC', init_w=INIT_W)

            # fc_out = L.linear(
            #     out_dim=self.n_code*2, layer_dict=self.layers,
            #     inputs=cnn_out, init_w=INIT_W, wd=self._wd, name='Linear')

            return fc_out

    def sample_latent(self):
        with tf.variable_scope('sample_latent'):
            cnn_out = self.layers['encoder_out']
            
            z_mean = L.linear(
                out_dim=self.n_code, layer_dict=self.layers,
                inputs=cnn_out, init_w=INIT_W, wd=self._wd, name='latent_mean')
            z_std = L.linear(
                out_dim=self.n_code, layer_dict=self.layers, nl=L.softplus,
                inputs=cnn_out, init_w=INIT_W, wd=self._wd, name='latent_std')
            z_log_std = tf.log(z_std + 1e-8)

            b_size = tf.shape(cnn_out)[0]
            z = ops.tf_sample_diag_guassian(z_mean, z_std, b_size, self.n_code)
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

            return tf.tanh(decoder_out)

    def discriminator(self, inputs):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            fc_out = modules.discriminator_FC(inputs, self.is_training,
                                              nl=L.leaky_relu,
                                              wd=self._wd, name='discriminator_FC',
                                              init_w=INIT_W)
            return fc_out

    def sample_prior(self):
        b_size = tf.shape(self.image)[0]
        samples = ops.tf_sample_standard_diag_guassian(b_size, self.n_code)
        return samples

    def _get_loss(self):
        with tf.name_scope('reconstruction_loss'):
            # with tf.name_scope('likelihood'):
            p_hat = self.layers['decoder_out']
            p = self.image
            # cross_entropy = p * tf.log(p_hat + 1e-6) + (1 - p) * tf.log(1 - p_hat + 1e-6)
            # cross_entropy_loss = -tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=[1,2,3]))
            autoencoder_loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(p - p_hat), axis=[1,2,3]))
            # autoencoder_loss = tf.reduce_mean(tf.square(p - p_hat))
            # with tf.name_scope('GAN'):
            #     gan_loss = tf.nn.sigmoid_cross_entropy_with_logits(
            #         labels=tf.ones_like(self.layers['fake']),
            #         logits=self.layers['fake'],
            #         name='gan_loss')
            #     self.gan_loss = tf.reduce_mean(gan_loss)

            return autoencoder_loss * self._enc_w

    def _get_optimizer(self):
        return tf.train.AdamOptimizer(self.lr, beta1=0.5)
        # return tf.train.MomentumOptimizer(self.lr, momentum=0.9)

    def get_train_op(self):
        with tf.name_scope('train'):
            opt = self.get_optimizer()
            loss = self.get_loss()
            var_list = tf.trainable_variables(scope='AE')
            print(var_list)
            grads = tf.gradients(loss, var_list)
            return opt.apply_gradients(zip(grads, var_list))

    def get_generator_train_op(self):
        with tf.name_scope('generator_train_op'):
            with tf.name_scope('generator_loss'):
                gan_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(self.layers['fake']),
                    logits=self.layers['fake'],
                    name='output')
                self.gan_loss = tf.reduce_mean(gan_loss)
            opt = tf.train.AdamOptimizer(self.lr, beta1=0.5)
            # opt = tf.train.MomentumOptimizer(self.lr, momentum=0.1)
            # var_list = tf.trainable_variables(scope=['AE/encoder'])
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='AE/encoder') +\
                       tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='AE/sample_latent')
            print(var_list)
            grads = tf.gradients(self.gan_loss * self._gen_w, var_list)
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
                d_loss = tf.reduce_mean(loss_real) + tf.reduce_mean(loss_fake)
                self.d_loss = d_loss
                
            opt = tf.train.AdamOptimizer(self.lr, beta1=0.5)
            # opt = tf.train.MomentumOptimizer(self.lr, momentum=0.1)
            # dc_var = [var for var in all_variables if 'dc_' in var.name]
            var_list = tf.trainable_variables(scope='discriminator')
            # print(tf.trainable_variables())
            print(var_list)
            grads = tf.gradients(self.d_loss * self._gen_w, var_list)
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
            tf.cast(self.layers['sample_im'], tf.float32),
            collections=['train'])
        tf.summary.histogram(
            name='real distribution', values=self.real_distribution,
            collections=['train'])
        tf.summary.histogram(
            name='encoder distribution', values=self.layers['z'],
            collections=['train'])

        # var_list = tf.trainable_variables()
        # [tf.summary.histogram('gradient/' + var.name, grad, 
        #  collections=['train']) for grad, var in zip(grads, var_list)]
        
        return tf.summary.merge_all(key='train')

