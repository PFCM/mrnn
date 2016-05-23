"""Test out the guys at approximating sequences"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from mrnn import tensor_ops as tops


def get_tt_model(inputs, shape, initial_states,
                 nonlinearity=tf.identity,
                 trainable=True, name='TT'):
    """Gets a recurrent tensor train model.
    
    Args:
      inputs: list of tensors -- the input sequence in time major order.
        (ie. a list of tensors each of shape `[batch_size x input_size]`.
      shape: the shape of the model (how many layers, the shape of each
        layer. Should be a sequence of sequences, one for each layer and
        each one consisting of (input_size, output_size, hidden_size, 
        rank1, rank2).
      initial_states: initial hidden states. Should be one per layer of
        appropriate size.
      trainable: whether the variables are added to the tensorflow
        collection of trainable variables (and hence whether they will
        be optimised or not).
    
    Returns:
      list of tensors: the outputs of the net.
    """
    with tf.variable_scope(name) as scope:
        states = [initial_states]
        for step, net_in in enumerate(inputs):
            if step == 1:
                scope.reuse_variables()
            output = net_in
            # do the layers
            layer_states = []
            for i, layer in enumerate(shape):
                tensor = tops.get_tt_3_tensor(layer[:3], layer[3:],
                                              trainable=trainable,
                                              name='layer_{}_W'.format(i+1))
                output = tops.bilinear_product_tt_3(output, tensor,
                                                    states[step][i])
                output = nonlinearity(output)
                # add output to states
                layer_states.append(output)
            states.append(layer_states)
        outputs = [step_states[-1] for step_states in states[1:]]
    return outputs


def mean_squared_error(values, targets):
    """Mean squared error across a whole sequence.

    Args:
      values: sequence of tensors
      targets: sequence of tensors.
    """
    if len(values) != len(targets):
        raise ValueError('sequence lengths need to match up')

    return tf.reduce_sum(tf.square(tf.pack(values)-tf.pack(targets)))


def get_train_op(loss):
    """Gets an op to minimized the given loss. Uses everything in 
    tensorflow's trainable variable collection.

    Args:
      loss: the tensor to be minimised.
    """
    opt = tf.train.GradientDescentOptimizer(0.0001)
    return opt.minimize(loss)


def get_data(num, producer_inputs, producer_outputs, sess):
    inputs = []
    targets = []
    for _ in range(num):
        results = sess.run(producer_inputs + producer_outputs)
        inputs.append(results[:len(results)//2])
        targets.append(results[len(results)//2:])
    return inputs, targets


def run_epoch(num_batches, prod_inputs, prod_outputs, targets, model_inputs,
              outputs, train_op, loss, sess):
    """generates num_batches worth of data and then runs on them,
    returning the average loss"""
    inputs, labels = get_data(num_batches, prod_inputs, prod_outputs,
                              sess)
    total_loss = 0
    for in_data, t_data in zip(inputs, labels):
        # fill the dict
        feed = {}
        for in_var, input_ in zip(model_inputs, in_data):
            feed[in_var] = input_
        for t_var, label in zip(targets, t_data):
            feed[t_var] = label
        # do a batch
        batch_loss, _ = sess.run([loss, train_op],
                                 feed)
        total_loss += batch_loss
        print('\r----loss: {:.5f}'.format(batch_loss), end='')
    print()
    return total_loss / num_batches


if __name__ == '__main__':
    tf.set_random_seed(0xabcd)
    num_in = 110
    num_out = 100
    state_size = 200
    net_shape = [[num_in, 256, state_size, 20, 20],
                 [256, num_out, state_size, 20, 20]]
    seq_length = 75
    batch_size = 32
    # first get models, one producer and one to try approx it
    with tf.variable_scope('producer',
                           initializer=tf.random_normal_initializer(
                               stddev=0.1,
                               seed=1)):
        prod_inputs = [tf.random_uniform([batch_size, num_in], minval=-1.0, maxval=1.0)] * seq_length
        producer_states = tf.unpack(tf.ones([len(net_shape), batch_size, state_size]))
        prod_outs = get_tt_model(prod_inputs, net_shape, producer_states,
                                 trainable=False)
        
    with tf.variable_scope('model',
                           initializer=tf.random_normal_initializer(
                               stddev=0.5,
                               seed=1)):
        inputs = [tf.placeholder(tf.float32, name='input_{}'.format(i),
                                 shape=[batch_size, num_in])
                  for i in range(seq_length)]
        targets = [tf.placeholder(tf.float32, name='target_{}'.format(i),
                                  shape=[batch_size, num_out])
                   for i in range(seq_length)]
        initial_states = tf.unpack(tf.ones([len(net_shape), batch_size, state_size]))
        outputs = get_tt_model(inputs, net_shape, initial_states,
                               trainable=True,
                               nonlinearity=tf.nn.tanh)

    loss = mean_squared_error(targets, outputs)
    train_op = get_train_op(loss)

    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.initialize_all_variables())
        print('baseline error: {}'.format(
            run_epoch(100, prod_inputs, prod_outs, targets, inputs,
                      outputs, tf.no_op(), loss, sess)))
        for epoch in range(100):
            error = run_epoch(500, prod_inputs, prod_outs, targets, inputs,
                              outputs, train_op, loss, sess)
            print('Epoch {}: error: {}'.format(epoch+1, error))
