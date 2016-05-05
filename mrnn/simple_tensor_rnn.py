"""A very simple rnn that uses some tensors.
Hidden state and output look like:

math::
    h_{t+1} = \rho(h_t W x_t + h_t^T U + Vx_t + b)

Where W is a 3-tensor. This amounts to just doing the bilinear
product, but augmenting the h and x with 1s and correspondingly
expanding the tensor in all dimensions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from mrnn.handy_ops import *
from mrnn.tensor_ops import *


class SimpleRandomSparseCell(tf.nn.rnn_cell.RNNCell):
    """Implements the above with a random sparse W.
    """

    def __init__(self, num_units, num_inputs, sparsity,
                 nonlinearity=tf.nn.relu):
        self._num_units = num_units
        self._num_inputs = num_inputs
        self._nonlinearity = nonlinearity
        self._sparsity = sparsity

    @property
    def sparsity(self):
        return self._sparsity

    @property
    def state_size(self):
        return self._num_units

    @property
    def input_size(self):
        return self._num_inputs

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, states, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # this is the mode-3 matricization of W :)
            big_tensor = random_sparse_tensor(
                [self._num_units,
                 self._num_inputs * self._num_units],
                self.sparsity, name='W_3')
            u = tf.get_variable('U', [self._num_units, self._num_units])
            v = tf.get_variable('V', [self._num_units, self._num_inputs])
            b = tf.get_variable('b', [self._num_units],
                                initializer=tf.constant_initializer(0.0))
            # make and flatten the outer product
            # have to do this with some unfortunate reshaping
            outer_prod = tf.batch_matmul(
                tf.reshape(states, [-1, self._num_units, 1]),
                tf.reshape(inputs, [-1, 1, self._num_inputs]))
            outer_prod = tf.reshape(
                outer_prod,
                [-1, self._num_units * self._num_inputs])
            tensor_prod = tf.sparse_tensor_dense_matmul(
                big_tensor, outer_prod, adjoint_b=True)
            tensor_prod = tf.transpose(tensor_prod)
            hidden_act = tf.matmul(states, u)
            input_act = tf.matmul(inputs, v)
            linears = tensor_prod + hidden_act
            linears += input_act
            linears += b
            output = self._nonlinearity(linears)
            return output, output


class SimpleRandomSparseCell2(tf.nn.rnn_cell.RNNCell):
    """As above, but with a more careful implementation"""
    pass


class SimpleCPCell(tf.nn.rnn_cell.RNNCell):
    """Super simple net, but with a tensor stored in its
    CP approximation of given rank."""

    def __init__(self, num_units, num_inputs, rank,
                 nonlinearity=tf.nn.relu, weightnorm=False):
        self._num_units = num_units
        self._num_inputs = num_inputs
        self._rank = rank
        self._nonlinearity = nonlinearity
        self._weightnorm = weightnorm

    @property
    def rank(self):
        return self._rank

    @property
    def state_size(self):
        return self._num_units

    @property
    def input_size(self):
        return self._num_inputs

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, states, scope=None):
        """does the stuff"""
        with tf.variable_scope(scope or type(self).__name__):
            # first we need to get the tensor
            tensor = get_cp_tensor([self._num_units+1,
                                    self._num_units+1,
                                    self._num_inputs+1],
                                   self._rank,
                                   'W_hat',
                                   weightnorm=self._weightnorm)
            vec_a = tf.concat(
                1, [inputs, tf.ones([inputs.get_shape()[0].value, 1])])
            vec_b = tf.concat(
                1, [states, tf.ones([inputs.get_shape()[0].value, 1])])
            result = bilinear_product_cp(vec_a, tensor, vec_b)[:,:self._num_units]
            result = self._nonlinearity(result)
            return result, result
