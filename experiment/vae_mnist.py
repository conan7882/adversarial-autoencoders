#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vae_mnist.py
# Author: Qian Ge <geqian1001@gmail.com>

import os
import sys
import numpy as np
import tensorflow as tf
import platform
import scipy.misc
import argparse
import matplotlib.pyplot as plt

sys.path.append('../')
from src.dataflow.mnist import MNISTData
from src.models.vae import VAE
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
    SAVE_PATH = '/home/qge2/workspace/data/out/vae/vae/'
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
    parser.add_argument('--load', type=int, default=99,
                        help='Load step of pre-trained')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Init learning rate')
    parser.add_argument('--ncode', type=int, default=2,
                        help='number of code')

    parser.add_argument('--bsize', type=int, default=128,
                        help='Init learning rate')
    parser.add_argument('--maxepoch', type=int, default=100,
                        help='Max iteration')

    
    return parser.parse_args()


def preprocess_im(im):
    im = im / 255.
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

    with tf.variable_scope('VAE') as scope:
        model = VAE(n_code=FLAGS.ncode, wd=0)
        model.create_train_model()

    with tf.variable_scope('VAE') as scope:
        scope.reuse_variables()
        valid_model = VAE(n_code=FLAGS.ncode, wd=0)
        valid_model.create_generate_model(b_size=400)

    trainer = Trainer(model, valid_model, train_data, init_lr=FLAGS.lr, save_path=SAVE_PATH)
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
            trainer.train_epoch(sess, summary_writer=writer)
            trainer.valid_epoch(sess, summary_writer=writer)
            if epoch_id % 10 == 0:
                saver.save(sess, '{}vae-epoch-{}'.format(SAVE_PATH, epoch_id))
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
    valid_data = MNISTData('test',
                            data_dir=DATA_PATH,
                            shuffle=True,
                            pf=preprocess_im,
                            batch_dict_name=['im', 'label'])
    batch_data = valid_data.next_batch_dict()
    plt.figure()
    plt.imshow(np.squeeze(batch_data['im'][0]))
    plt.show()
    print(batch_data['label'])

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

