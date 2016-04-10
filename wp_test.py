"""Set up a very quick model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cProfile

import numpy as np
import tensorflow as tf

import rnndatasets.warandpeace as data
import mrnn

flags = tf.flags
flags.DEFINE_float('learning_rate', 0.0001, 'the learning rate')
flags.DEFINE_integer('num_steps', 100, 'how far to back propagate in time')
flags.DEFINE_integer('batch_size', 100, 'how many batches')
flags.DEFINE_integer('width', 200, 'how many units per layer')
flags.DEFINE_integer('num_layers', 3, 'how many layers')
flags.DEFINE_integer('num_epochs', 100, 'how many times through the data')
flags.DEFINE_integer('num_chars', 1000000, 'how much of the data to use')
flags.DEFINE_boolean('profile', False, 'if true, runs the whole lot in a'
                                       'profiler')

FLAGS = flags.FLAGS


def inference(input_var, shape, vocab_size, num_steps):
    """Makes the model up to logits.

    Args:
        input_var: variable to hold the inputs.
        shape: list of layer sizes.
        vocab_size: the number of symbols in and out.
        num_steps: the length of the sequences.

    Returns:
        (outputs, state) where outputs is a big tensor of output logits,
            post softmax projection (but before actual softmax)
            and state is the final state of the rnn.
    """
    # first thing we need is some kind of embedding
    embedding = tf.get_variable('embedding', [vocab_size, shape[0]])
    inputs = tf.nn.embedding_lookup(embedding, input_var)
    # set up the cells
    last_size = shape[0]
    cells = []
    for layer in shape:
        cells.append(mrnn.MRNNCell(layer, last_size))
        last_size = layer
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)
    # make sure the inputs are an approprite list
    inputs = [tf.squeeze(input_, [1])
              for input_ in tf.split(1, num_steps, inputs)]
    # and make the rnn
    outputs, state = tf.nn.rnn(cell, inputs, dtype=tf.float32)
    # push all the outputs together
    outputs = tf.reshape(tf.concat(1, outputs), [-1, layer])
    # get softmax weights and bias
    softmax_w = tf.get_variable("softmax_w", [last_size, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    logits = tf.matmul(outputs, softmax_w) + softmax_b
    return logits, state


def loss(logits, targets, vocab_size, batch_size, num_steps):
    """Gets the loss scalar.

    Args:
        logits: the raw output of the network, a big of tensors
        targets: the targets they should be getting close to
            (as an int tensor).

    Returns:
        scalar tensor representing the average cross entropy.
    """
    # first we have to softmax it
    cost = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(targets, [-1])],
        [tf.ones([batch_size * num_steps])])
    cost = tf.reduce_mean(cost)
    return cost


def train(cost, learning_rate, max_grad_norm=5.0):
    """Gets the training op"""
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      max_grad_norm)
    opt = tf.train.AdamOptimizer(learning_rate)
    return opt.apply_gradients(zip(grads, tvars))


def fill_feed(batch, input_var, target_var):
    """Fills the feed dict with a batch (including setting up targets)"""
    inputs = batch[:-1]
    targets = batch[1:]
    return {
        input_var: inputs,
        target_var: targets
    }


def run_epoch(sess, data_iter, initial_state, final_state, cost, train_op,
              input_var, target_var):
    """Runs an epoch, pulling from data_iter until it is empty."""

    # get into it
    costs = 0
    steps = 0
    for progress, batch in data_iter:
        batch_loss, _ = sess.run([cost, train_op],
                                 feed_dict=fill_feed(batch,
                                                     input_var,
                                                     target_var))
        costs += batch_loss
        steps += 1
        print('\r({:.3f}) -- xent: {:.4f}'.format(progress, costs/steps), end='')
    print()


def gen_sample(vocab):
    """Gets a sample from the network"""
    pass


def main(_):
    """do the stuff"""
    # first we have to get the model
    print('...getting model...', end='')
    inputs = tf.placeholder(tf.int32,
                            [FLAGS.batch_size, FLAGS.num_steps])
    targets = tf.placeholder(tf.int32,
                             [FLAGS.batch_size, FLAGS.num_steps])

    vocab = data.get_vocab('char')

    with tf.variable_scope('rnn_model') as scope:
        full_outputs, final_state = inference(
            inputs, [FLAGS.width] * FLAGS.num_layers,
            len(vocab), FLAGS.num_steps)
        av_cost = loss(full_outputs, targets, len(vocab),
                       FLAGS.batch_size, FLAGS.num_steps)
        train_op = train(av_cost, FLAGS.learning_rate)
    print('\r{:~^30}'.format('got model'))
    sess = tf.Session()
    with sess.as_default():
        print('...initialising...', end='')
        sess.run(tf.initialize_all_variables())
        print('\r{:~^30}'.format('initialised'))
        print('\n\n{:~>30}'.format('training: '))
        for epoch in range(FLAGS.num_epochs):
            print('~~Epoch: {}'.format(epoch+1))
            data_iter = data.get_char_iter(FLAGS.num_steps+1,
                                           FLAGS.batch_size,
                                           report_progress=True,
                                           overlap=1,
                                           sequential=True)
            run_epoch(sess, data_iter, None, final_state, av_cost, train_op,
                      inputs, targets)

if __name__ == '__main__':
    if FLAGS.profile:
        cProfile.run('tf.app.run()')
    else:
        tf.app.run()
