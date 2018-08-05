#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: aae_mnist.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import sys
import numpy as np
import tensorflow as tf
import platform
import scipy.misc
import argparse
# import matplotlib.pyplot as plt

sys.path.append('../')
from src.dataflow.mnist import MNISTData
from src.models.vae import VAE
from src.models.aae import AAE
from src.helper.trainer import Trainer
from src.helper.generator import Generator
from src.helper.visualizer import Visualizer
import src.models.distribution as distribution

if platform.node() == 'Qians-MacBook-Pro.local':
    DATA_PATH = '/Users/gq/workspace/Dataset/MNIST_data/'
    SAVE_PATH = '/Users/gq/tmp/draw/'
    RESULT_PATH = '/Users/gq/tmp/ram/center/result/'
elif platform.node() == 'arostitan':
    DATA_PATH = '/home/qge2/workspace/data/MNIST_data/'
    SAVE_PATH = '/home/qge2/workspace/data/out/vae/'
else:
    DATA_PATH = 'E://Dataset//MNIST//'
    SAVE_PATH = 'E:/tmp/vae/'
    # RESULT_PATH = 'E:/tmp/ram/trans/result/'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--generate', action='store_true',
                        help='generate')
    parser.add_argument('--viz', action='store_true',
                        help='visualize')
    parser.add_argument('--test', action='store_true',
                        help='test')
    parser.add_argument('--label', action='store_true',
                        help='use label for training')
    parser.add_argument('--load', type=int, default=99,
                        help='Load step of pre-trained')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Init learning rate')
    parser.add_argument('--ncode', type=int, default=2,
                        help='number of code')

    parser.add_argument('--bsize', type=int, default=128,
                        help='Init learning rate')
    parser.add_argument('--maxepoch', type=int, default=100,
                        help='Max iteration')

    parser.add_argument('--encw', type=float, default=1.,
                        help='weight of encoder loss')
    parser.add_argument('--genw', type=float, default=6.,
                        help='weight of generator loss')
    parser.add_argument('--disw', type=float, default=6.,
                        help='weight of discriminator loss')

    parser.add_argument('--dist', type=str, default='gaussian',
                        help='prior')
    
    return parser.parse_args()


def preprocess_im(im):
    im = im / 255. * 2. - 1.
    return im

def train():
    FLAGS = get_args()
    train_data = MNISTData('train',
                            data_dir=DATA_PATH,
                            shuffle=True,
                            pf=preprocess_im,
                            batch_dict_name=['im', 'label'])
    train_data.setup(epoch_val=0, batch_size=FLAGS.bsize)
    valid_data = MNISTData('test',
                            data_dir=DATA_PATH,
                            shuffle=True,
                            pf=preprocess_im,
                            batch_dict_name=['im', 'label'])
    valid_data.setup(epoch_val=0, batch_size=FLAGS.bsize)

    model = AAE(n_code=FLAGS.ncode, wd=0, use_label=FLAGS.label,
                enc_weight=FLAGS.encw, gen_weight=FLAGS.genw, dis_weight=FLAGS.disw)
    model.create_train_model()

    valid_model = AAE(n_code=FLAGS.ncode, wd=0)
    valid_model.create_generate_model(b_size=400)

    trainer = Trainer(model, valid_model, train_data, use_label=FLAGS.label,
                      init_lr=FLAGS.lr, save_path=SAVE_PATH)
    if FLAGS.ncode == 2:
        z = distribution.interpolate(plot_size=20)
        z = np.reshape(z, (400, 2))
        visualizer = Visualizer(model, save_path=SAVE_PATH)
    else:
        z = None
    generator = Generator(generate_model=valid_model, save_path=SAVE_PATH)

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        writer = tf.summary.FileWriter(SAVE_PATH)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        for epoch_id in range(FLAGS.maxepoch):
            trainer.train_gan_epoch(sess, distr_type=FLAGS.dist, summary_writer=writer)
            trainer.valid_epoch(sess, summary_writer=writer)
            
            if epoch_id % 10 == 0:
                saver.save(sess, '{}aae-epoch-{}'.format(SAVE_PATH, epoch_id))
                if FLAGS.ncode == 2:
                    generator.generate_samples(sess, plot_size=20, z=z, file_id=epoch_id)
                    visualizer.viz_2Dlatent_variable(sess, valid_data, file_id=epoch_id)
                

def generate():
    FLAGS = get_args()
    plot_size = 20

    with tf.variable_scope('VAE') as scope:
        # scope.reuse_variables()
        generate_model = VAE(n_code=FLAGS.ncode, wd=0)
        generate_model.create_generate_model(b_size=plot_size*plot_size)

    generator = Generator(generate_model=generate_model, save_path=SAVE_PATH)

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '{}vae-epoch-{}'.format(SAVE_PATH, FLAGS.load))
        generator.generate_samples(sess, plot_size=plot_size, z=None)

def visualize():
    FLAGS = get_args()
    plot_size = 20

    valid_data = MNISTData('test',
                            data_dir=DATA_PATH,
                            shuffle=True,
                            pf=preprocess_im,
                            batch_dict_name=['im', 'label'])
    valid_data.setup(epoch_val=0, batch_size=FLAGS.bsize)

    with tf.variable_scope('VAE') as scope:
        model = VAE(n_code=FLAGS.ncode, wd=0)
        model.create_train_model()

    with tf.variable_scope('VAE') as scope:
        scope.reuse_variables()
        valid_model = VAE(n_code=FLAGS.ncode, wd=0)
        valid_model.create_generate_model(b_size=400)

    visualizer = Visualizer(model, save_path=SAVE_PATH)
    generator = Generator(generate_model=valid_model, save_path=SAVE_PATH)

    z = distribution.interpolate(plot_size=plot_size)
    z = np.reshape(z, (plot_size*plot_size, 2))

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, '{}vae-epoch-{}'.format(SAVE_PATH, FLAGS.load))
        visualizer.viz_2Dlatent_variable(sess, valid_data)
        generator.generate_samples(sess, plot_size=plot_size, z=z)

def test():

    import matplotlib.pyplot as plt
    # import src.models.ops as ops
    # samples = ops.tf_sample_standard_diag_guassian(12800, 2)
    real_sample = distribution.gaussian_mixture(
        12800, 2, n_labels=10, x_var=0.5, y_var=0.1, label_indices=None)

    plt.figure()
    plt.scatter(real_sample[:,0], real_sample[:,1], s=3)
    plt.show()

    # valid_data = MNISTData('test',
    #                         data_dir=DATA_PATH,
    #                         shuffle=True,
    #                         pf=preprocess_im,
    #                         batch_dict_name=['im', 'label'])
    # batch_data = valid_data.next_batch_dict()
    # plt.figure()
    # plt.imshow(np.squeeze(batch_data['im'][0]))
    # plt.show()
    # print(batch_data['label'])

if __name__ == '__main__':
    FLAGS = get_args()

    if FLAGS.train:
        train()
    elif FLAGS.generate:
        generate()
    elif FLAGS.viz:
        visualize()
    elif FLAGS.test:
        test()

