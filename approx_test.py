"""Test to see if our methods can learn to approximate some arbitrary tensor.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf

import mrnn.tensor_ops as tops

logger = logging.getLogger(__name__)


def generate_data(tensor, num):
    """Generates some data given a tensor.
    Inputs are between -1 and 1, range of outputs depends on tensor.

    Doing this in numpy as currently wayy too slooww.
    Args:
        tensor: a numpy (3-way) tensor to use.
        num: how many points to generate

    Returns:
        (a, b, targets)
    """
    a_features = tensor.shape[0]
    b_features = tensor.shape[2]
    a = np.random.sample(size=(num, a_features)) * 2 - 1
    b = np.random.sample(size=(num, b_features)) * 2 - 1
    outputs = np.einsum('ij,jkl,ml->ik', a, tensor, b)
    return a, b, outputs


def get_affine_model(input_a, input_b, shape):
    """Gets a series of affine transforms (ie a linear neural net).
    Inputs are concatenated and biases are applied (hence affine).

    Total number of parameters is prod(shape) + sum(shape).

    Args:
        input_a: the first input (we expect batch major, so [batch x features])
        input_b: the second input variable.
        shape: list of shapes for the matrices. Last one will be the shape
            of the outputs.

    Returns:
        outputs
    """
    def layer(inputs, size, name='layer'):
        """makes a layer"""
        with tf.name_scope(name):
            weights = tf.get_variable(name+'_W', [inputs.get_shape()[1], size])
            biases = tf.get_variable(name+'_b', [size])
            return tf.nn.bias_add(tf.matmul(inputs, weights), biases)
    last_input = tf.concat(1, [input_a, input_b])
    for i, size in enumerate(shape):
        last_input = layer(last_input, size, name='layer-{}'.format(i))

    return last_input


def get_cp_model(input_a, input_b, output_size, rank):
    """Gets a cp approximation of an appropriately shaped tensor

    Args:
        input_a: the first input variable (batch major)
        input_b: the second input
        output_size: the middle dimensions of the tensor we are approximating.
        rank: controls the number of parameters.
    """
    a_size = input_a.get_shape()[1].value
    b_size = input_b.get_shape()[1].value
    tensor = tops.get_cp_tensor([a_size, output_size, b_size], rank, 'tensor')
    return tops.bilinear_product_cp(input_a, tensor, input_b)


def get_sparse_model(input_a, input_b, output_size, sparsity):
    """Gets a model which uses a randomly sparse tensor.

    Args:
        input_a: the first input variable (batch major)
        input_b: the second input
        output_size: the middle dimensions of the tensor we are approximating.
        rank: controls the number of parameters.
    """
    a_size = input_a.get_shape()[1].value
    b_size = input_b.get_shape()[1].value
    tensor = tops.get_sparse_tensor([a_size, output_size, b_size], sparsity)
    return tops.bilinear_product_sparse(input_a, tensor, input_b, output_size)


def mean_squared_error(a, b):
    """gets the mse between two tensors"""
    return tf.reduce_mean(tf.squared_difference(a, b))


def get_training_op(loss):
    """Returns an op to minimise the given loss"""
    opt = tf.train.RMSPropOptimizer(1e-2)
    return opt.minimize(loss)


def batch_sequence(batch_size, *items):
    """iterates a sequence of numpy arrays"""
    num_batches = items[0].shape[0] // batch_size

    for batch_num in range(num_batches):
        print('\r({:.4f})'.format((batch_num+1)/num_batches), end='',
              flush=True)
        yield [item[batch_num*batch_size:(batch_num+1)*batch_size, ...]
               for item in items]


def run_epoch(sess, data, placeholders, batch_size, train_op, loss_op):
    """runs on some data, reports back the average loss"""
    a_var, b_var, t_var = placeholders
    total_loss = 0
    steps = 0
    for a, b, targets in batch_sequence(batch_size, *data):
        loss, _ = sess.run([loss_op, train_op],
                           {a_var: a,
                            b_var: b,
                            t_var: targets})
        print(' loss: {:.6f}'.format(loss), end='', flush=True)
        total_loss += loss
        steps += 1
    return total_loss / steps


def get_placeholders(batch_size, a_size, b_size, target_size):
    """get the placeholders we need"""
    a_var = tf.placeholder(tf.float32, [batch_size, a_size], name='input_a')
    b_var = tf.placeholder(tf.float32, [batch_size, b_size], name='input_b')
    t_var = tf.placeholder(tf.float32, [batch_size, target_size],
                           name='targets')

    return a_var, b_var, t_var


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    a_size = 5
    b_size = 5
    c_size = 5
    batch_size = 10
    np.random.seed(18121991)
    tensor = np.random.sample((a_size, c_size, b_size))

    print('getting model')
    a_var, b_var, t_var = get_placeholders(batch_size, a_size, b_size, c_size)

    with tf.variable_scope('model',
                           initializer=tf.random_normal_initializer(
                                stddev=0.15)):
        # model_outs = get_affine_model(a_var, b_var, [15, 15, c_size])
        # model_outs = get_cp_model(a_var, b_var, 5, 20)
        model_outs = get_sparse_model(a_var, b_var, 5, .1)
        loss_op = mean_squared_error(model_outs, t_var)
        train_op = get_training_op(loss_op)

    print('starting to train')
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    summaries = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('summaries', sess.graph)
    with sess.as_default():
        for epoch in range(100):
            # print('Epoch {}'.format(epoch+1))
            #print('getting training data')
            a, b, c = generate_data(tensor, 500)
            av_loss = run_epoch(sess, (a, b, c), (a_var, b_var, t_var),
                                batch_size, train_op, loss_op)
            # print('\n  train loss: {}'.format(av_loss))
        print()
        # get some unseen data
        a, b, c = generate_data(tensor, 1000)
        av_loss = run_epoch(sess, (a, b, c), (a_var, b_var, t_var),
                            batch_size, tf.no_op(), loss_op)
        print('\ntest loss: {}'.format(av_loss))
