"""Let's see if a tensor layer can learn to multiply"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf

import mrnn.tensor_ops


def get_batch(batch_size, binary=False):
    """Gets a batch of data.

    Args:
        batch_size (int): how much data to generate.
        binary (Optional[bool]): whether the data should be scalars in (0,1]
            or binary vectors (8 bit, with 16 bit results).

    Returns:
        batch: (a, b, target) where target = ab (elementwise).
    """
    if not binary:
        # eas
        input_a = np.random.random((batch_size, 1))
        input_b = np.random.random((batch_size, 1))
        target = input_a * input_b
    else:
        input_a = np.random.randint(256, size=(batch_size, 1))
        input_b = np.random.randint(256, size=(batch_size, 1))
        target = input_a * input_b
        input_a = np.unpackbits(input_a.astype(np.uint8))
        input_b = np.unpackbits(input_b.astype(np.uint8))
        # now do target
        target_lsb = target & 0xff
        target_lsb = np.unpackbits(target_lsb.astype(np.uint8))
        target_msb = target >> 8
        target_msb = np.unpackbits(target_msb.astype(np.uint8))
        target = np.hstack((target_msb.reshape(batch_size, 8),
                            target_lsb.reshape(batch_size, 8)))
    return input_a, input_b, target


def layer(in_size, input_var, out_size, name='layer'):
    """does a single layer"""
    weights = tf.get_variable(name+'_W', [in_size, out_size])
    bias = tf.get_variable(name+'_b', [out_size])
    return tf.nn.relu(tf.matmul(input_var, weights) + bias)


def get_feedforward_model(input_var, shape, binary=False):
    """Gets an mlp with relus.

    Args:
        input_var: placeholder for inputs
        shape: list of layer shapes
        target_var: target outputs

    Returns:
        output
    """
    # go through each size in shape and make the layer
    layer_input = input_var
    last_size = input_var.get_shape()[1].value
    # set initialiser?
    for i, size in enumerate(shape):
        layer_input = layer(last_size, layer_input, size,
                            'layer-{}'.format(i+1))
        if size != shape[-1]:
            layer_input = tf.nn.relu(layer_input)
        last_size = size

    if binary:
        return tf.nn.sigmoid(layer_input)

    return tf.nn.relu(layer_input)


def mean_square_error(value, target):
    """Returns mse between value and target."""
    return tf.reduce_mean(tf.square(value-target))


def train(loss):
    """Gets an op to do a step toward minimising the loss"""
    # opt = tf.train.GradientDescentOptimizer(0.01)
    tvars = tf.trainable_variables()
    gnorm = tf.reduce_sum(tf.abs(tvars[0]))
    for tvar in tvars[1:]:
        gnorm += tf.reduce_sum(tf.abs(tvar))
    loss += 0.1 * gnorm
    # opt = tf.train.MomentumOptimizer(0.001, 0.99)
    opt = tf.train.RMSPropOptimizer(0.01)
    return opt.minimize(loss)


def evaluate_model(input_a, input_b, target, batch_size, model,
                   num_steps=50000):
    """does the stuff for real"""
    model_out, model_loss, train_op = model
    # for now super simple, just run a given number of steps and test
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    with sess.as_default():
        for step in range(num_steps):
            data_a, data_b, targets = get_batch(batch_size)
            step_loss, _ = sess.run([model_loss, train_op],
                                    {input_a: data_a,
                                     input_b: data_b,
                                     target: targets})
            if (step + 1) % 100 == 0:
                print('\r--step: {} error: {:-<30}'.format(step+1, step_loss),
                      end='', flush=True)
        print()
        # and run on some test data
        test_loss = 0
        for step in range(num_steps//10):
            data_a, data_b, targets = get_batch(batch_size)
            step_loss, = sess.run([model_loss],
                                  {input_a: data_a,
                                   input_b: data_b,
                                   target: targets})
            test_loss += step_loss
        print('Test MSE: {}'.format(test_loss/(num_steps//10)))
        # see if it generalises to a new range of numbers
        test_loss = 0
        for step in range(num_steps//10):
            data_a, data_b, targets = get_batch(batch_size)
            data_a = data_a * 2
            data_b = data_b * 2
            targets = targets * 4
            step_loss, = sess.run([model_loss],
                                  {input_a: data_a,
                                   input_b: data_b,
                                   target: targets})
            test_loss += step_loss
        print('Generalisation MSE: {}'.format(test_loss/(num_steps//10)))


def bias_pad(tensor):
    """pads a one to the end of a batch of tensors"""
    return tf.concat(1, (tensor, tf.ones([tensor.get_shape()[0].value, 1])))


def get_cptensor_model(input_a, input_b, max_rank, num_out, binary=False):
    """gets a cp factorised tensor model"""
    if not binary:
        model = mrnn.tensor_ops.get_cp_tensor([2, num_out, 2], max_rank, 't')
    else:
        model = mrnn.tensor_ops.get_cp_tensor([9, num_out, 9], max_rank, 't')
    pad_a = bias_pad(input_a)
    pad_b = bias_pad(input_b)
    output = mrnn.tensor_ops.bilinear_product_cp(pad_a, model, pad_b)
    # output = tf.nn.relu(output)
    output = layer(num_out, output, 1)
    if binary:
        return tf.nn.sigmoid(output)
    return output


def cheat(input_a, input_b):
    return input_a * input_b


def main():
    """do things"""
    # try and learn to multiply
    tf.set_random_seed(int.from_bytes(b'batcave', 'big'))
    batch_size = 256
    input_a = tf.placeholder(tf.float32, name='input_a', shape=[batch_size, 1])
    input_b = tf.placeholder(tf.float32, name='input_b', shape=[batch_size, 1])
    all_ins = tf.concat(1, (input_a, input_b))
    targets = tf.placeholder(tf.float32, name='targets', shape=[batch_size, 1])

    print('doing large mlp')
    with tf.variable_scope('mlp',
                           initializer=tf.random_normal_initializer(
                                stddev=0.01)):
        # let's make it stupidly big?
        ff_net_out = get_feedforward_model(all_ins,
                                           [20, 1])
        ff_net_loss = mean_square_error(ff_net_out, targets)
        ff_net_train = train(ff_net_loss)

    evaluate_model(input_a, input_b, targets, batch_size,
                   (ff_net_out, ff_net_loss, ff_net_train))
    print('doing tensor')
    with tf.variable_scope('tensor',
                           initializer=tf.random_normal_initializer(
                                stddev=2)):
        tnet_out = get_cptensor_model(input_a, input_b, 2, 2)
        tnet_loss = mean_square_error(tnet_out, targets)
        tnet_train = train(tnet_loss)

    evaluate_model(input_a, input_b, targets, batch_size,
                   (tnet_out, tnet_loss, tnet_train))
    print('cheating')
    with tf.variable_scope('tensor',
                           initializer=tf.random_normal_initializer(
                                stddev=.01)):
        cheat_out = cheat(input_a, input_b)
        cheat_loss = mean_square_error(cheat_out, targets)

    evaluate_model(input_a, input_b, targets, batch_size,
                   (cheat_out, cheat_loss, tf.no_op()))


if __name__ == '__main__':
    main()
