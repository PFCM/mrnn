"""Test some guys on penn treebank by word?"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import shutil

from six.moves import xrange

import numpy as np
import tensorflow as tf

import mrnn
import mrnn.init as init
import rnndatasets.ptb as ptb

flags = tf.app.flags

# model parameters
flags.DEFINE_string('cell', 'vanilla', 'what cell to use')
flags.DEFINE_bool('embed', False, 'whether or not to use word embeddings')
flags.DEFINE_integer('width', 200, 'the width of each layer')
flags.DEFINE_integer('layers', 2, 'how many hidden layers')
flags.DEFINE_integer('rank', 50, 'rank of the tensor decomposition, if using')

# training parameters -- defaults are as per the dropout LSTM paper except with adam
flags.DEFINE_float('learning_rate', 0.01, 'base learning rate for ADAM')
flags.DEFINE_integer('batch_size', 20, 'minibatch size')
flags.DEFINE_integer('sequence_length', 35, 'how far we unroll BPTT')
flags.DEFINE_float('grad_clip', 10000.0, 'where to clip the gradients')
# flags.DEFINE_integer('start_decay', 6, 'when to start the learning rate decay')
# flags.DEFINE_float('decay_factor', 1.2, 'how much to divide the learning rate'
#                    'by each epoch after start_decay')
flags.DEFINE_integer('num_epochs', 15, 'how long to train for')
flags.DEFINE_float('dropout', 1.0, 'how much dropout (if at all)')
flags.DEFINE_integer('reset_steps', 0, 'how often to reset the state during training')
flags.DEFINE_float('epsilon', 1e-8, '`a small constant for numerical stability`')


# housekeeping
flags.DEFINE_string('results_dir', None, 'where to store the results')
flags.DEFINE_integer('seed', 1001, 'seed for the random numbers')

FLAGS = flags.FLAGS


def fill_batch(input_vars, target_vars, data):
    """makes a feed dict"""
    feed = {}
    for vars, np_data in zip((input_vars, target_vars), data):
        for var, value in zip(vars, np_data.T):
            feed[var] = value
    return feed


def run_epoch(sess, data_iter, initial_state, final_state,
              cost, train_op, input_vars, target_vars,
              reset_after=0, grad_norm=None):
    """Runs an epoch of training"""
    costs = 0
    steps = 0
    gnorm = 0
    state = initial_state.eval(session=sess)
    for batch in data_iter:
        feed_dict = fill_batch(input_vars, target_vars, batch)
        if reset_after > 0 and steps % reset_after == 0:
            state = initial_state.eval(session=sess)
        feed_dict[initial_state] = state
        if grad_norm is None:
            batch_loss, state, _ = sess.run(
                [cost, final_state, train_op],
                feed_dict=feed_dict)

            costs += batch_loss
            steps += 1
            if steps % 10 == 0:
                print('\r...({}) - xent: {}'.format(steps, costs/steps), end='')
        else:
            batch_loss, state, _, batch_gnorm = sess.run(
                [cost, final_state, train_op, grad_norm],
                feed_dict=feed_dict)
            gnorm += batch_gnorm
            costs += batch_loss
            steps += 1
            if steps % 10 == 0:
                print(
                    '\r...({}) - xent: {} (g norm {})'.format(
                        steps, costs/steps, gnorm/steps),
                    end='', flush=True)
    print('..epoch over')
    if grad_norm is None:
        return costs/steps
    return costs/steps, gnorm/steps


def get_placeholders(batch_size, sequence_length):
    """Get the placeholders required to feed into an RNN.

    Args:
        batch_size: size of the batches.
        sequence_length: how much we are unrolling

    Returns:
        inputs, targets: both sequence_length lists of tensors.
            They will both be int32 with shape
            `[batch_size]`.
    """
    shape = [batch_size]
    dtype = tf.int32
    inputs = [tf.placeholder(dtype, shape=shape, name='input{}'.format(i))
              for i in xrange(sequence_length)]
    targets = [tf.placeholder(dtype, shape=shape, name='target{}'.format(i))
               for i in xrange(sequence_length)]
    return inputs, targets


def get_cell(input_size, hidden_size):
    """Gets a cell with given params, according to FLAGS.cell"""
    if FLAGS.cell == 'cp+-':
        return mrnn.AddSubCPCell(hidden_size, input_size, FLAGS.rank)
    elif FLAGS.cell == 'cp+':
        return mrnn.AdditiveCPCell(hidden_size, input_size, FLAGS.rank)
    elif FLAGS.cell == 'cp-del':
        return mrnn.CPDeltaCell(hidden_size, input_size, FLAGS.rank, weightnorm='partial')
    elif FLAGS.cell == 'simple_cp':
        return mrnn.SimpleCPCell(hidden_size, input_size, FLAGS.rank, nonlinearity=tf.nn.tanh)
    elif FLAGS.cell == 'cp-loss':
        return mrnn.CPLossyIntegrator(hidden_size, input_size, FLAGS.rank)
    elif FLAGS.cell == 'lstm':
        return tf.nn.rnn_cell.BasicLSTMCell(hidden_size, input_size=input_size,
                                           state_is_tuple=False)
    elif FLAGS.cell == 'vanilla':
        return mrnn.VRNNCell(hidden_size, input_size=input_size,
                             hh_init=init.orthonormal_init(0.5))
    elif FLAGS.cell == 'vanilla-weightnorm':
        return mrnn.VRNNCell(hidden_size, input_size=input_size,
                             weightnorm='recurrent',
                             hh_init=init.orthonormal_init(0.5))
    elif FLAGS.cell == 'vanilla-layernorm':
        return mrnn.VRNNCell(hidden_size, input_size=input_size,
                             weightnorm='layer',
                             hh_init=init.orthonormal_init(0.5))
    elif FLAGS.cell == 'irnn':
        return mrnn.IRNNCell(hidden_size, input_size=input_size)
    else:
        raise ValueError('unknown cell: {}'.format(FLAGS.cell))


def inference(inputs, shape, vocab_size, dropout=1.0):
    """Build a model.

    Args:
        inputs: list of int32 input tensors containing the ids.
        shape: shape of the network (list of ints).
        vocab_size: size of the input vocabulary.
        sequence_length: how far to unroll the network.
        dropout: how much dropout to apply in between hidden layers.

    Returns:
        outputs, final_state, initial_state:
            - outputs is a list of `[batch_size, vocab]` float32 tensors
                containing the (linear) outputs of the network.
            - final_state is a tensor containing the final hidden states of the
                network.
            - initial_state is a tensor which will contain the initial state
                of the network.
    """
    # first we are going to want to get the one-hot inputs
    # biiigg -- could make a non-trainable identity matrix and do embedding
    # lookup?
    one_hot_inputs = [tf.one_hot(step, vocab_size) for step in inputs]
    batch_size = inputs[0].get_shape()[0].value

    cells = [get_cell(vocab_size, shape[0])]

    for i, layer in enumerate(shape[1:]):
        cells.append(get_cell(shape[i-1], layer))
    if dropout != 1.0:
        cells = [tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=dropout)
                 for cell in cells]
    cell = tf.nn.rnn_cell.MultiRNNCell(cells)

    init_state = cell.zero_state(batch_size, tf.float32)
    outputs, state = tf.nn.rnn(cell, one_hot_inputs, initial_state=init_state,
                               dtype=tf.float32)
    # we have to now project the outputs
    softmax_w = tf.get_variable('softmax_w', [shape[-1], vocab_size])
    softmax_b = tf.get_variable('softmax_b', [vocab_size])
    # probably faster to cat them all, do an enormous matmul and reshape?
    logits = [tf.nn.bias_add(tf.matmul(output, softmax_w), softmax_b)
              for output in outputs]
    return logits, state, init_state


def loss(logits, targets):
    """gets the loss scalar.

    Args:
        logits: a big list of tensors, the output of the network at each step.
        targets: the targets (as ints) -- another list for each step

    Returns:
        scalar tensor: average cross-entropy per word.
    """
    cost = tf.nn.seq2seq.sequence_loss_by_example(
        logits,
        targets,
        [tf.ones_like(target, dtype=tf.float32) for target in targets])  # equal weighting
    return tf.reduce_mean(cost)


def get_train_op(cost, learning_rate, max_grad_norm=1000.0, global_step=None):
    """gets a training op (ADAM)"""
    opt = tf.train.AdamOptimizer(learning_rate, epsilon=FLAGS.epsilon)
    # opt = tf.train.GradientDescentOptimizer(learning_rate)
    grads_and_vars = opt.compute_gradients(cost)
    grads, norm = tf.clip_by_global_norm([grad for grad, var in grads_and_vars],
                                         max_grad_norm)
    t_op = opt.apply_gradients(
        [(grad, var) for grad, (_, var) 
         in zip(grads, grads_and_vars)],
        global_step=global_step)
    return t_op, norm


def write_params(results_dir):
    """Counts the number of parameters in tf.trainable_variables()
    and writes the number to a file in the results directory called
    `params.txt`.
    """
    filename = os.path.join(results_dir, 'params.txt')
    tvars = tf.trainable_variables()
    total = 0
    for var in tvars:
        prod = 1
        for dim in var.get_shape().as_list():
            prod *= dim
        total += prod
    print('~~~Model has {} params.'.format(total))
    with open(filename, 'w') as fp:
        fp.write('{} trainable parameters.\n'.format(total))


def main(_):
    """Do things"""
    tf.set_random_seed(FLAGS.seed)
    print('\n~\n~~\n~~~\n...getting data...')
    train, valid, test, vocab = ptb.get_ptb_data()
    print('\n~\n~~\n~~~\n...getting model...', end='')
    inputs, targets = get_placeholders(FLAGS.batch_size, FLAGS.sequence_length)

    if FLAGS.dropout != 1.0:
        dropout = tf.get_variable('dropout', [], trainable=False,
                                  initializer=tf.constant_initializer(
                                      FLAGS.dropout))
    else:
        dropout = 1.0

    global_step = tf.Variable(0, name='global_step')

    with tf.variable_scope('rnn_model') as scope:
        full_outputs, final_state, init_state = inference(
            inputs, [FLAGS.width] * FLAGS.layers,
            len(vocab), dropout=dropout)
    print('\r{:~^60}'.format('got model'))

    # get the training stuff
    with tf.variable_scope('training'):
        av_cost = loss(full_outputs, targets)

        lr_var = tf.Variable(FLAGS.learning_rate,
                             name='learning_rate',
                             trainable=False)
        train_op, grad_norm = get_train_op(av_cost, lr_var,
                                           max_grad_norm=FLAGS.grad_clip,
                                           global_step=global_step)
        print('{:~^60}'.format('(got training ops)'))
    lr = FLAGS.learning_rate

    saver = tf.train.Saver(tf.trainable_variables(),
                           max_to_keep=3)
    model_dir = os.path.join(FLAGS.results_dir,
                             'models')
    model_name = os.path.join(model_dir,
                              FLAGS.cell)
    model_name += '({})'.format(
        '-'.join([str(FLAGS.width)] * FLAGS.layers))

    tv_filename = os.path.join(FLAGS.results_dir,
                               'training.txt')
    test_filename = os.path.join(FLAGS.results_dir,
                                 'test.txt')
    
    best_valid_loss = np.inf
    best_model_path = None
    best_model_dir = os.path.join(model_dir, 'best')
    os.makedirs(best_model_dir, exist_ok=True)
    
    # count and report the params
    write_params(FLAGS.results_dir)

    sess = tf.Session()
    with sess.as_default():
        print('..initialising..', end='', flush=True)
        sess.run(tf.initialize_all_variables())
        print('\r{:~^60}'.format('initialised'))

        for epoch in xrange(FLAGS.num_epochs):
            print('~~Epoch: {}'.format(epoch+1))
            if dropout != 1.0:
                sess.run(dropout.assign(FLAGS.dropout))
            epoch_loss, avgnorm = run_epoch(
                sess, ptb.batch_iterator(train,
                                         FLAGS.batch_size,
                                         FLAGS.sequence_length),
                init_state, final_state,
                av_cost, train_op,
                inputs, targets, grad_norm=grad_norm,
                reset_after=FLAGS.reset_steps)
            print('~~~~training perp: {}'.format(np.exp(epoch_loss)))
            # ditch dropout
            if dropout != 1.0:
                sess.run(dropout.assign(1.0))
            valid_loss = run_epoch(sess, ptb.batch_iterator(valid,
                                                            FLAGS.batch_size,
                                                            FLAGS.sequence_length),
                                   init_state, final_state,
                                   av_cost, tf.no_op(),
                                   inputs, targets)
            print('~~~~valid perp: {}'.format(np.exp(valid_loss)))
            # save, write
            steps = global_step.eval()
            path = saver.save(sess, model_name, global_step=steps,
                              write_meta_graph=False)
            if valid_loss < best_valid_loss:
                print('{:~^60}'.format('new record'))
                best_model_path = shutil.copy(path, best_model_dir)
                best_valid_loss = valid_loss
            else:
                print('{:~^60}'.format('no improvement'))

            with open(tv_filename, 'a') as fp:
                fp.write('{}, {}, {}\n'.format(epoch_loss, avgnorm, valid_loss))
        # load the best model
        print('~Loading best model: {}'.format(best_model_path))
        saver.restore(sess, best_model_path)
        # get the test loss
        test_loss = run_epoch(
            sess, ptb.batch_iterator(test, FLAGS.batch_size, FLAGS.sequence_length),
            init_state, final_state, av_cost, tf.no_op(), inputs, targets)
        with open(test_filename, 'w') as fp:
            fp.write('Test cross-entropy: {}'.format(test_loss))


if __name__ == '__main__':
    tf.app.run()
