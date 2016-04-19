"""A very simple rnn that uses some tensors.
Hidden state and output look like: 

math::
    h_{t+1} = \rho(h_t W x_t + V x_t + b)

Where W is a 3-tensor.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from mrnn.handy_ops import *


class SimpleRandomSparseCell(tf.nn.rnn_cell.RNNCell):
    """Implements the above with a random sparse W.
    """

    def __init__(self, num_units, num_inputs, sparsity, nonlinearity=tf.nn.relu):
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

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # this is the mode-3 matricization of W
            big_tensor = random_sparse_tensor([self._num_inputs * self._num_units, self._num_units],
                                              self.sparsity, name='W')
            input_weights = tf.get_variable('V', [self._num_inputs, self._num_units])
            bias = tf.get_variable('b', [self._num_units])

            # (num_inputs*num_units x num_units) x (batch_size x num_units)T
            # = (num_inputs*num_units x batch_size)
            step_a = tf.sparse_tensor_dense_matmul(
                big_tensor, state, adjoint_b=True)
            # reshape to (num_units x num_inputs x batch_size) and then
            # multiply each mode 1 slice with the inputs
            input_shape = [shape.value for shape in inputs.get_shape()]
            step_a = tf.reshape(step_a, [self._num_units, self._num_inputs, input_shape[0]])
            step_b = tf.batch_matmul(
                tf.tile(tf.reshape(inputs, [1, input_shape[0], input_shape[1]]),
                        [self._num_units, 1, 1]),
                step_a)
            # not sure about this at all
            step_b = tf.reduce_sum(step_b, reduction_indices=[1])
            step_b = tf.transpose(step_b)
            
            proj_inputs = tf.matmul(inputs, input_weights)

            output = self._nonlinearity(step_b + proj_inputs + bias)
            return output, output
