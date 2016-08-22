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
                 ' (', progressbar.DynamicMessage('nll'), ')',
                 '{', progressbar.ETA(), '}'],
        redirect_stdout=True)


def fill_feed(input_vars, target_vars, data):
    feed = {}
    for vars, np_data in zip((input_vars, target_vars), data):
        for var, value in zip(vars, np_data.transpose([1, 0, 2])):
            feed[var] = value
    return feed


def negative_log_likelihood(logits, targets):
    """Gets the NLL the logits assign to targets, assuming they're all
    to be independently sigmoided."""
    # if we rescale the targets so off is -1 and on is 1
    # then we can multiply through the logits
    # and sigmoid gives us the probabilities :)
    # because 1-sigmoid(x) = sigmoid(-x)
    targets = [(2.0 * targ) - 1.0 for targ in targets]
    probs = [tf.sigmoid(logit * targ) for logit, targ in zip(logits, targets)]
    probs = [tf.reduce_sum(tf.log(prob), reduction_indices=1)
             for prob in probs]
    return -tf.reduce_mean(tf.pack(probs))


def count_params():
    """count trainable variables"""
    tvars = tf.trainable_variables()
    total = 0
    for var in tvars:
        prod = 1
        for dim in var.get_shape().as_list():
            prod *= dim
        total += prod
    return total


def main(_):
    tf.set_random_seed(FLAGS.seed)
    # start by getting the data
    print('...getting data', end='')
    train, valid, test = jsb.get_data()

    inputs, targets = get_placeholders(FLAGS.batch_size,
                                       FLAGS.sequence_length)
    print('\r{:\\^60}'.format('got data'))

    print('...getting model', end='')

    if FLAGS.weight_noise != 0.0:
        weight_noise = tf.get_variable(
            'weight_noise', trainable=False,
            initializer=FLAGS.weight_noise,
            collections=[tf.GraphKeys.LOCAL_VARIABLES])
    else:
        weight_noise = 0.0

    with tf.variable_scope('rnn'):
        all_outputs, final_state, init_state = ptb_test.inference(
            inputs, [FLAGS.width] * FLAGS.layers, jsb.NUM_FEATURES,
            weight_noise=weight_noise)
    print('\r{:/^60}'.format('got model'))

    global_step = tf.Variable(0, name='global_step', trainable=False)

    print('...getting training ops', end='')
    with tf.variable_scope('training'):
        loss_op = sigmoid_xent(all_outputs, targets)
        nll_op = negative_log_likelihood(all_outputs, targets)

        train_op, grad_norm = ptb_test.get_train_op(
            nll_op, FLAGS.learning_rate, max_grad_norm=FLAGS.grad_clip,
            global_step=global_step)
    print('\r{:\\^60}'.format('got train ops'))
    print('{:/^60}'.format('{} params'.format(count_params())))
    # do saving etc
    saver = tf.train.Saver(
        tf.trainable_variables() + [global_step], max_to_keep=1)
    # make sure we have somewhere to write
    os.makedirs(FLAGS.results_dir, exist_ok=True)

    def test_model(all_data):
        """convenience for gathering statistics"""
        nll = 0
        xent = 0
        step = 0
        for data in jsb.batch_iterator(all_data, FLAGS.batch_size, FLAGS.sequence_length):
            feed = fill_feed(inputs, targets, data)

            batch_xent, batch_nll = sess.run([loss_op, nll_op],
                                             feed_dict=feed)

            xent += batch_xent
            nll += batch_nll
            step += 1
        return xent/step, nll/step

    sess = tf.Session()
    with sess.as_default():
        print('..initialising', end='')
        sample_weights = mrnn.merge_variational_initialisers()
        sess.run(tf.initialize_local_variables())
        if sample_weights is not None:
            print('(initialising variational wrappers)')
            sess.run(sample_weights)
        sess.run(tf.initialize_all_variables())
        print('\r{:\\^60}'.format('initialised'))

        bar = get_progressbar()
        num_steps = train.shape[0] * FLAGS.num_epochs // (FLAGS.batch_size * FLAGS.sequence_length)

        bar.start(num_steps)

        best_valid_nll = 100000
        best_model_path = ''
        for epoch in xrange(FLAGS.num_epochs):
            if weight_noise != 0.0:
                sess.run(weight_noise.assign(FLAGS.weight_noise))
            for data in jsb.batch_iterator(train, FLAGS.batch_size, FLAGS.sequence_length):
                feed = fill_feed(inputs, targets, data)
                if sample_weights is not None:
                    sess.run(sample_weights)
                batch_xent, batch_nll, _ = sess.run([loss_op, nll_op, train_op],
                                                     feed_dict=feed)
                step = global_step.eval()
                bar.update(step, nll=batch_nll)

            if weight_noise != 0.0:
                sess.run(weight_noise.assign(0.0))
            if sample_weights is not None:
                sess.run(sample_weights)
            valid_xent, valid_nll = test_model(valid)

            print('Epoch {}'.format(epoch+1))
            print('~~ valid xent: {}'.format(valid_xent))
            print('~~ valid  nll: {}'.format(valid_nll))

            if valid_nll < best_valid_nll:
                print('~~ (new record)')
                best_model_path = saver.save(sess,
                                             os.path.join(FLAGS.results_dir,
                                                          FLAGS.cell + '.checkpoint'),
                                             write_meta_graph=False,
                                             global_step=step)
                best_valid_nll = valid_nll

        bar.finish()
        # load best model and do test
        print('~~ Loading best model for final results')
        saver.restore(sess, best_model_path)
        test_xent, test_nll = test_model(test)
        valid_xent, valid_nll = test_model(valid)
        train_xent, train_nll = test_model(train)
        print('Test  xent: {}, nll: {}'.format(test_xent, test_nll))
        print('Valid xent: {}, nll: {}'.format(valid_xent, valid_nll))
        print('Train xent: {}, nll: {}'.format(train_xent, train_nll))

        results_filename = os.path.join(FLAGS.results_dir, 'earlystopped_results.txt')
        with open(results_filename, 'w') as fp:
            fp.write('Test  xent: {}, nll: {}\n'.format(test_xent, test_nll))
            fp.write('Valid xent: {}, nll: {}\n'.format(valid_xent, valid_nll))
            fp.write('Train xent: {}, nll: {}\n'.format(train_xent, train_nll))


if __name__ == '__main__':
    tf.app.run()
