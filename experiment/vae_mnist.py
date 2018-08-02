#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: vae_mnist.py
# Author: Qian Ge <geqian1001@gmail.com>

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
from src.helper.trainer import Trainer
from src.helper.generator import Generator
import src.models.distribution as distribution

if platform.node() == 'Qians-MacBook-Pro.local':
    DATA_PATH = '/Users/gq/Google Drive/Foram/CNN Data/code/GAN/MNIST_data/'
    SAVE_PATH = '/Users/gq/tmp/draw/'
    RESULT_PATH = '/Users/gq/tmp/ram/center/result/'
elif platform.node() == 'arostitan':
    DATA_PATH = '/home/qge2/workspace/data/MNIST_data/'
    SAVE_PATH = '/home/qge2/workspace/data/out/draw/'
else:
    DATA_PATH = 'E://Dataset//MNIST//'
    SAVE_PATH = 'E:/tmp/draw/'
    # RESULT_PATH = 'E:/tmp/ram/trans/result/'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--test', action='store_true',
                        help='Test')
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
    thr = 0.7
    im[np.where(im < thr)] = 0
    im[np.where(im > 0)] = 1
    return im

def train():
    FLAGS = get_args()
    train_data = MNISTData('train',
                            data_dir=DATA_PATH,
                            shuffle=True,
                            pf=preprocess_im,
                            batch_dict_name=['im', 'label'])
    train_data.setup(epoch_val=0, batch_size=FLAGS.bsize)

    with tf.variable_scope('VAE') as scope:
        model = VAE(n_code=FLAGS.ncode, wd=0)
        model.create_train_model()

        train_op = model.get_train_op()
        loss_op = model.get_loss()
        train_summary_op = model.get_train_summary()

    with tf.variable_scope('VAE') as scope:
        scope.reuse_variables()
        valid_model = VAE(n_code=FLAGS.ncode, wd=0)
        valid_model.create_generate_model(b_size=400)

        valid_summary_op = valid_model.get_valid_summary()

    trainer = Trainer(model, valid_model, train_data, init_lr=FLAGS.lr, save_path=SAVE_PATH)
    z = distribution.interpolate(n_code=20)
    z = np.reshape(z, (400, 2))
    generator = Generator(generate_model=valid_model, save_path=SAVE_PATH)

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        writer = tf.summary.FileWriter(SAVE_PATH)
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)

        for epoch_id in range(FLAGS.maxepoch):
            trainer.train_epoch(sess, summary_writer=writer)
            trainer.valid_epoch(sess, batch_size=128, summary_writer=writer)
            generator.random_generate(sess, batch_size=128, z=z)


def generate():
    FLAGS = get_args()
    z = distribution.interpolate(n_code=20)
    print(z.shape)
    z = np.reshape(z, (400, 2))
    with tf.variable_scope('VAE') as scope:
        # scope.reuse_variables()
        generate_model = VAE(n_code=FLAGS.ncode, wd=0)
        generate_model.create_generate_model(b_size=400)

    generator = Generator(generate_model=generate_model, save_path=SAVE_PATH)

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = True
    with tf.Session(config=sessconfig) as sess:
        sess.run(tf.global_variables_initializer())
        generator.random_generate(sess, batch_size=128, z=z)

if __name__ == '__main__':
    train()
    # generate()
    # FLAGS = get_args()

    # model = DRAW(im_channel=1,
    #              n_encoder_hidden=256,
    #              n_decoder_hidden=256,
    #              n_code=10,
    #              im_size=28,
    #              n_step=10)

    # if FLAGS.train:
    #     train_data = MNISTData('train',
    #                            data_dir=DATA_PATH,
    #                            shuffle=True,
    #                            pf=preprocess_im,
    #                            batch_dict_name=['im'])
    #     train_data.setup(epoch_val=0, batch_size=128)

    #     model.create_model()
    #     train_op = model.train_op()
    #     loss_op = model.get_loss()

    #     model.create_generate_model(b_size=10)
    #     generate_op = model.layers['generate']
    #     summary_op = model.get_summary()

    #     writer = tf.summary.FileWriter(SAVE_PATH)
    #     decoder_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='step/decoder')
    #     write_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='step/write')
    #     saver = tf.train.Saver(
    #             var_list=decoder_var + write_var)
    #     sessconfig = tf.ConfigProto()
    #     sessconfig.gpu_options.allow_growth = True
    #     with tf.Session(config=sessconfig) as sess:
    #         sess.run(tf.global_variables_initializer())
    #         writer.add_graph(sess.graph)
    #         loss_sum = 0
    #         for i in range(0, 10000):
    #             batch_data = train_data.next_batch_dict()

    #             _, loss = sess.run([train_op, loss_op],
    #                                feed_dict={model.raw_image: batch_data['im'],
    #                                           model.lr: FLAGS.lr})

    #             loss_sum += loss

    #             if i % 100 == 0:
    #                 print(loss_sum / 100.)
    #                 loss_sum = 0
    #                 cur_summary = sess.run(summary_op)
    #                 writer.add_summary(cur_summary, i)
    #                 saver.save(sess, '{}draw_step_{}'.format(SAVE_PATH, i))

    # if FLAGS.test:
    #     model.create_generate_model(b_size=10)
    #     generate_op = model.layers['generate']
    #     step_op = model.layers['gen_step'] 

    #     decoder_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='step/decoder')
    #     write_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='step/write')
    #     saver = tf.train.Saver(
    #             var_list=decoder_var + write_var)
    #     sessconfig = tf.ConfigProto()
    #     sessconfig.gpu_options.allow_growth = True
    #     with tf.Session(config=sessconfig) as sess:
    #         sess.run(tf.global_variables_initializer())
    #         saver.restore(sess, '{}draw_step_{}'.format(SAVE_PATH, 9900))
    #         gen_im, gen_step = sess.run([generate_op, step_op])

    #         for step_id, step_batch in enumerate(gen_step):
    #             for idx, im in enumerate(step_batch):
    #                 scipy.misc.imsave('{}im_{}_step_{}.png'.format(SAVE_PATH, idx, step_id), np.squeeze(im))

    # # trainer = Trainer(model, train_data, init_lr=FLAGS.lr)
    # # writer = tf.summary.FileWriter(SAVE_PATH)
    # # saver = tf.train.Saver()

    # # sessconfig = tf.ConfigProto()
    # # sessconfig.gpu_options.allow_growth = True
    # # with tf.Session(config=sessconfig) as sess:
    # #     sess.run(tf.global_variables_initializer())
    # #     if FLAGS.train:
    # #         writer.add_graph(sess.graph)
    # #         for step in range(0, config.epoch):
    # #             trainer.train_epoch(sess, summary_writer=writer)
    # #             trainer.valid_epoch(sess, valid_data, config.batch)

    # #             saver.save(sess, '{}ram-{}-mnist-step-{}'.format(SAVE_PATH, name, config.step), global_step=step)
    # #     if FLAGS.predict:
    # #         valid_data.setup(epoch_val=0, batch_size=20)
    # #         saver.restore(sess, '{}ram-{}-mnist-step-6-{}'.format(SAVE_PATH, name, FLAGS.load))
            
    # #         batch_data = valid_data.next_batch_dict()
    # #         trainer.test_batch(
    # #             sess,
    # #             batch_data,
    # #             unit_pixel=config.unit_pixel,
    # #             size=config.glimpse,
    # #             scale=config.n_scales,
    # #             save_path=RESULT_PATH)

    # #     if FLAGS.test:
    # #         train_data.setup(epoch_val=0, batch_size=2)
    # #         batch_data = train_data.next_batch_dict()
    # #         test, trans_im = sess.run(
    # #             [model.layers['retina_reprsent'], model.pad_im],
    # #             feed_dict={model.image: batch_data['data'],
    # #                        model.label: batch_data['label'],
    # #                        })
    # #         # print(test.shape)
    # #         tt = 0
    # #         for glimpse_i, trans, im in zip(test, trans_im, batch_data['data']):
    # #             scipy.misc.imsave('{}trans_{}.png'.format(SAVE_PATH, tt),
    # #                               np.squeeze(trans))
    # #             for idx in range(0, 3):
    # #                 scipy.misc.imsave('{}g_{}_{}.png'.format(SAVE_PATH, tt, idx), 
    # #                                   np.squeeze(glimpse_i[0,:,:,idx]))
    # #             scipy.misc.imsave('{}im_{}.png'.format(SAVE_PATH, tt), np.squeeze(im))
    # #             tt += 1
        

    #     # writer.close()
