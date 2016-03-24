"""TensorFlow implementation of http://arxiv.org/pdf/1312.6114v10.pdf
for photo restoration with only conv and deconv first"""

from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import prettytensor as pt  # https://github.com/google/prettytensor
import scipy.misc
import tensorflow as tf
from scipy.misc import imsave
import input_data
from deconv import deconv2d_hack
from ops import deconv2d
from progressbar import ETA, Bar, Percentage, ProgressBar

flags = tf.flags
logging = tf.logging
flags.DEFINE_integer("image_size", 64, "The size of image to use [64]")
flags.DEFINE_integer("batch_size", 1024, "batch size")
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 100, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "data/", "")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_integer("hidden_size", 2048, "size of the hidden VAE unit")  # prev 10,1142
FLAGS = flags.FLAGS

def encoder(input_tensor):
    '''Create encoder network.
    Args:
        input_tensor: a batch of flattened images [batch_size, 64*64]
    Returns:
        A tensor that expresses the encoder network
    '''
    return (pt.wrap(input_tensor).
            reshape([FLAGS.batch_size, FLAGS.image_size, FLAGS.image_size, 1]).
            conv2d(5, 32, stride=2).
            conv2d(5, 64, stride=2).
            conv2d(5, 128, edges='VALID').
            flatten()).tensor
    # fully_connected(FLAGS.hidden_size * 2, activation_fn=None)).tensor


def decoder(input_tensor=None):
    '''Create decoder network.
        If input tensor is provided then decodes it, otherwise samples from 
        a sampled vector.
    Args:
        input_tensor: a batch of vectors to decode
    Returns:
        A tensor that expresses the decoder network
    '''
    epsilon = tf.random_normal([FLAGS.batch_size, FLAGS.hidden_size])
    # output_tensor = tf.random_normal([FLAGS.batch_size, FLAGS.image_size,FLAGS.image_size,1])
    if input_tensor is None:
        mean = None
        stddev = None
        input_sample = epsilon
    else:
        mean = input_tensor[:, :FLAGS.hidden_size]
        stddev = tf.sqrt(tf.exp(input_tensor[:, :FLAGS.hidden_size]))
        input_sample = mean + epsilon * stddev

    # return tf.reshape(deconv2d((pt.wrap(input_sample).
    #         reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size]).
    #         deconv2d_hack(3, 128, edges='VALID').
    #         deconv2d_hack(3, 128, edges='VALID').
    #         deconv2d_hack(3, 128, edges='VALID').
    #         deconv2d_hack(5, 128, edges='VALID').
    #         deconv2d_hack(5, 64, edges='VALID').
    #         deconv2d_hack(3, 64, edges='VALID').
    #         deconv2d_hack(2, 64, edges='VALID').
    #         deconv2d_hack(5, 32, stride=2).
    #         deconv2d_hack(3, 32, stride=1).
    #         deconv2d_hack(2, 16, stride=2)
    #         ), [FLAGS.batch_size, FLAGS.image_size,FLAGS.image_size, 1]),
    #         shape=[FLAGS.batch_size, FLAGS.image_size*FLAGS.image_size*1]),mean, stddev
    return (pt.wrap(input_sample).
            reshape([FLAGS.batch_size, 1, 1, FLAGS.hidden_size]).
            deconv2d_hack(3, 128, edges='VALID').
            deconv2d_hack(3, 128, edges='VALID').
            deconv2d_hack(3, 128, edges='VALID').
            deconv2d_hack(3, 128, edges='VALID').
            deconv2d_hack(3, 128, edges='VALID').
            deconv2d_hack(2, 128, edges='VALID').
            deconv2d_hack(5, 64, edges='VALID').
            deconv2d_hack(5, 32, stride=2).
            deconv2d_hack(5, 1, stride=2, activation_fn=tf.nn.sigmoid).
            flatten()).tensor, mean, stddev


def get_vae_cost(mean, stddev, epsilon=1e-8):
    '''VAE loss
        See the paper
    Args:
        mean: 
        stddev:
        epsilon:
    '''
    return tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) -
                                2.0 * tf.log(stddev + epsilon) - 1.0))


def get_reconstruction_cost(output_tensor, target_tensor, epsilon=1e-8):
    '''Reconstruction loss
    Cross entropy reconstruction loss
    Args:
        output_tensor: tensor produces by decoder 
        target_tensor: the target tensor that we want to reconstruct
        epsilon:
    '''
    print (output_tensor.get_shape(),target_tensor.get_shape())
    print (tf.shape(output_tensor),tf.shape(target_tensor))
    return tf.reduce_sum(-target_tensor * tf.log(output_tensor + epsilon) -
                         (1.0 - target_tensor) * tf.log(1.0 - output_tensor + epsilon))


if __name__ == "__main__":
    #prep
    data_directory = os.path.join(FLAGS.working_directory, "celebACropped")
    if not os.path.exists(FLAGS.checkpoint_dir):os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(data_directory): os.makedirs(data_directory)
    celebACropped = input_data.read_data_sets(data_directory)

    #build model
    input_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size * FLAGS.image_size])
    ground_truth_tensor = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.image_size * FLAGS.image_size])
    with pt.defaults_scope(activation_fn=tf.nn.elu,
                           batch_normalize=True,
                           learned_moments_update_rate=0.0003,
                           variance_epsilon=0.001,
                           scale_after_normalization=True):
        with pt.defaults_scope(phase=pt.Phase.train):
            with tf.variable_scope("model") as scope:
                output_tensor, mean, stddev = decoder(encoder(input_tensor))
        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("model", reuse=True) as scope:
                sampled_tensor, _, _ = decoder()
                # sampled_tensor, _, _ = decoder(encoder(input_tensor))

    vae_loss = get_vae_cost(mean, stddev)
    rec_loss = get_reconstruction_cost(output_tensor, ground_truth_tensor)
    loss = vae_loss + rec_loss
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate, epsilon=1.0)
    train = pt.apply_optimizer(optimizer, losses=[loss])
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    #run as session
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(FLAGS.max_epoch):
            training_loss = 0.0
            widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
            pbar = ProgressBar(FLAGS.updates_per_epoch, widgets=widgets)
            pbar.start()
            for i in range(FLAGS.updates_per_epoch):
                pbar.update(i)
                x_masked, x_ground_truth = celebACropped.train.next_batch(FLAGS.batch_size)
                # x_masked, x_ground_truth = tf.to_float(x_masked), tf.to_float(x_ground_truth)
                _, loss_value = sess.run([train, loss],
                                         feed_dict={input_tensor: x_masked,ground_truth_tensor:x_ground_truth})
                training_loss += loss_value

            training_loss = training_loss / \
                            (FLAGS.updates_per_epoch * FLAGS.image_size * FLAGS.image_size * FLAGS.batch_size)
            print("Loss %f" % training_loss)

            if epoch%5 ==0:
                print("reached %5, save and evaluate results")
                saver.save(sess,save_path=os.path.join(FLAGS.checkpoint_dir, 'vae'),global_step=epoch)
                print(sampled_tensor.get_shape())
                imgs = sess.run(sampled_tensor)
                # imgs = sess.run(feed_dict={input_tensor: x_masked,ground_truth_tensor:x_ground_truth})
                for k in range(FLAGS.batch_size):
                    imgs_folder = os.path.join(FLAGS.working_directory, 'imgs'+str(epoch))
                    if not os.path.exists(imgs_folder): os.makedirs(imgs_folder)
                    imsave(os.path.join(imgs_folder, '%d.png') % k,
                           imgs[k].reshape(FLAGS.image_size, FLAGS.image_size))