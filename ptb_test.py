"""Test some guys on penn treebank by word"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os

from six.moves import xrange

import numpy as np
import tensorflow as tf

import mrnn
import wp_test
import rnndatasets.ptb as ptb

FLAGS = tf.app.flags.FLAGS


def run_epoch(sess, data_iter, initial_state, final_state,
              cost, train_op, input_var, target_var,
              reset_after=100):
    """Runs an epoch of training"""
    costs = 0
    steps = 0
    for batch in data_iter:
        if steps % reset_after == 0:
            state = initial_state.eval(session=sess)
        batch_loss, state, _ =sess.run(
            [cost, final_state, train_op],
            {input_var: batch[0],
             target_var: batch[1],
             initial_state: state})
        costs += batch_loss
        steps += 1
        print('\r...({}) - xent: {}'.format(steps, costs/steps), end='')
    print()
    return costs/steps
    

def main(_):
    """Do things"""
    tf.set_random_seed(FLAGS.seed)
    print('\n~\n~~\n~~~\n...getting data...')
    train, valid, test, vocab = ptb.get_ptb_data()
    print('\n~\n~~\n~~~\n...getting model...', end='')
    inputs = tf.placeholder(tf.int32,
                            [FLAGS.batch_size, FLAGS.num_steps])
    targets = tf.placeholder(tf.int32,
                             [FLAGS.batch_size, FLAGS.num_steps])
    # get a 1 step batch size 1 model for sampling
    inputs_1 = tf.placeholder(tf.int32, [1, 1])

    dropout = tf.get_variable('dropout', [], trainable=False)
    
    with tf.variable_scope('rnn_model') as scope:
        full_outputs, final_state, init_state = wp_test.inference(
            inputs, [FLAGS.width] * FLAGS.num_layers,
            len(vocab), FLAGS.num_steps, FLAGS.batch_size,
            dropout=dropout, return_initial_state=True)
        av_cost = wp_test.loss(full_outputs, targets, len(vocab),
                               FLAGS.batch_size, FLAGS.num_steps)
        lr_var = tf.get_variable('learning_rate', [], trainable=False)
        train_op = wp_test.train(av_cost, lr_var)
        scope.reuse_variables()
        outputs_1, state_1, init_state_1 = wp_test.inference(
            inputs_1, [FLAGS.width] * FLAGS.num_layers,
            len(vocab), 1, 1, return_initial_state=True)
        word_probs = tf.nn.softmax(outputs_1)
    print('\r{:~^60}'.format('got model'))

    lr = FLAGS.learning_rate

    saver = tf.train.Saver(tf.trainable_variables(),
                           max_to_keep=3)
    model_name = os.path.join(FLAGS.model_folder,
                              FLAGS.model_prefix)
    model_name += '({})'.format(
        '-'.join([str(FLAGS.width)] * FLAGS.num_layers))
    
    
    sess = tf.Session()
    with sess.as_default():
        print('..initialising..', end='', flush=True)
        sess.run(tf.initialize_all_variables())
        print('\r{:~^60}'.format('initialised'))

        for epoch in xrange(FLAGS.num_epochs):
            print('~~Epoch: {}'.format(epoch+1))
            if epoch % 10 == 0:
                samp = wp_test.gen_sample(vocab, word_probs,
                                          inputs_1, init_state_1,
                                          state_1, length=100)
                print(samp)
            sess.run(dropout.assign(FLAGS.dropout))
            epoch_loss = run_epoch(sess, ptb.batch_iterator(train,
                                                            FLAGS.batch_size,
                                                            FLAGS.num_steps),
                                   init_state, final_state,
                                   av_cost, train_op,
                                   inputs, targets)
            print('~~~~training perp: {}'.format(np.exp(epoch_loss)))
            # ditch dropout
            sess.run(dropout.assign(1.0))
            valid_loss = run_epoch(sess, ptb.batch_iterator(valid,
                                                            FLAGS.batch_size,
                                                            FLAGS.num_steps),
                                   init_state, final_state,
                                   av_cost, tf.no_op(),
                                   inputs, targets)
            print('~~~~valid perp: {}'.format(np.exp(valid_loss)))
            print('~' * 60)
                                   

  
if __name__ == '__main__':
    tf.app.run()
