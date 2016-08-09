"""Train some RNNS for sequential prediction on JSB"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import shutil

from six.moves import xrange

import numpy as np
import tensorflow as tf

import progressbar

import mrnn
import mrnn.init as init
import rnndatasets.jsbchorales as jsb

import ptb_test  # may as well reuse


FLAGS = tf.app.flags.FLAGS


def get_placeholders(batch_size, sequence_length):
    """Make input and target placeholders"""
    inputs = tf.placeholder(tf.float32, name='all_inputs',
                            shape=[sequence_length,
                                   batch_size,
                                   jsb.NUM_FEATURES])
    targets = tf.placeholder(tf.float32, name='all_targets',
                             shape=[sequence_length,
                                    batch_size,
                                    jsb.NUM_FEATURES])

    return tf.unpack(inputs), tf.unpack(targets)


def sigmoid_xent(logits, targets):
    """Gets the average cross entropy after having applied a sigmoid."""
    # reshape them into one huge batch
    logits = tf.concat(0, logits)
    targets = tf.concat(0, targets)

    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits, targets))


def get_progressbar():
    return progressbar.ProgressBar(
        widgets=['[', progressbar.Percentage(), '] ',
                 '(>₃>)｢ ',
                 progressbar.Bar(marker='~', left='', right='|'),
                 ' (', progressbar.DynamicMessage('xent'), ')',
                 '{', progressbar.ETA(), '}'],
        redirect_stdout=True)


def fill_feed(input_vars, target_vars, data):
    feed = {}
    for vars, np_data in zip((input_vars, target_vars), data):
        for var, value in zip(vars, np_data.transpose([1,0,2])):
            feed[var] = value
    return feed


def negative_log_likelihood(logits, targets):
    """Gets the NLL the logits assign to targets. Conveniently, this
    corresponds to the dot product of the target with the log-sigmoid
    activations at each step. We then average over timesteps and batches,
    this seems to give us numbers similar to everyone else."""
    logits = [tf.log(tf.sigmoid(logit)) for logit in logits]
    probs = [tf.batch_matmul(tf.expand_dims(step, 1),
                             tf.expand_dims(targ, 2))
             for step, targ in zip(logits, targets)]
    return -tf.reduce_mean(tf.pack(probs))


def main(_):
    tf.set_random_seed(FLAGS.seed)
    # start by getting the data
    print('...getting data', end='')
    train, valid, test = jsb.get_data()
    
    inputs, targets = get_placeholders(FLAGS.batch_size,
                                       FLAGS.sequence_length)
    print('\r{:\\^60}'.format('got data'))

    print('...getting model', end='')
    with tf.variable_scope('rnn'):
        all_outputs, final_state, init_state = ptb_test.inference(
            inputs, [FLAGS.width] * FLAGS.layers, jsb.NUM_FEATURES)
    print('\r{:/^60}'.format('got model'))

    global_step = tf.Variable(0, name='global_step', trainable=False)

    print('...getting training ops', end='')
    with tf.variable_scope('training'):
        loss_op = sigmoid_xent(all_outputs, targets)
        nll_op = negative_log_likelihood(all_outputs, targets)

        train_op, grad_norm = ptb_test.get_train_op(loss_op, FLAGS.learning_rate,
                                                    max_grad_norm=FLAGS.grad_clip,
                                                    global_step=global_step)
    print('\r{:\\^60}'.format('got train ops'))

    # do saving etc

    sess = tf.Session()
    with sess.as_default():
        print('..initialising', end='')
        sess.run(tf.initialize_all_variables())
        print('\r{:/^60}'.format('initialised'))

        bar = get_progressbar()
        num_steps = train.shape[0] * FLAGS.num_epochs // (FLAGS.batch_size * FLAGS.sequence_length)

        bar.start(num_steps)

        for epoch in xrange(FLAGS.num_epochs):
            for data in jsb.batch_iterator(train, FLAGS.batch_size, FLAGS.sequence_length):
                feed = fill_feed(inputs, targets, data)

                batch_xent, _ = sess.run([loss_op, train_op],
                                         feed_dict=feed)
                step = global_step.eval()
                bar.update(step, xent=batch_xent)

            valid_nll = 0
            valid_xent = 0
            valid_step = 0
            for data in jsb.batch_iterator(valid, FLAGS.batch_size, FLAGS.sequence_length):
                feed = fill_feed(inputs, targets, data)

                batch_xent, batch_nll = sess.run([loss_op, nll_op],
                                                 feed_dict=feed)

                valid_xent += batch_xent
                valid_nll += batch_nll
                valid_step += 1

            print('Epoch {}'.format(epoch+1))
            print('~~ valid xent: {}'.format(valid_xent/valid_step))
            print('~~ valid  nll: {}'.format(valid_nll/valid_step))
                

        bar.finish()
        

if __name__ == '__main__':
    tf.app.run()
