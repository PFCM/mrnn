"""Multiplicative RNN models. Special case of the tensor RNN"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf


class MRNNCell(tf.contrib.rnn.RNNCell):
    """Simple RNN but with a multiplicative rather than additive
    connection.
    """

    def __init__(self, num_units, input_size=None):
        """
        Set up a new MRNN model.

        Args:
            num_units(int): size of the hidden vector.
            input_size(Optional[int]): size of the input to this cell.
                if None, assumed to be the same as the number of hidden
                units.
        """
        self._num_units = num_units
        self._input_size = num_units if input_size is None else input_size

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def zero_state(self, batch_size, dtype):
        """Zero state is going to be all ones :)"""
        ones = tf.ones([batch_size, self.state_size], dtype=dtype)
        return ones

    def __call__(self, inputs, state, scope=None):
        """Gives us a multiplicative rnn:
            output = new_state = relu(Ux * Vh)
            where * is elementwise.
        """
        with tf.variable_scope(scope or type(self).__name__):
            hidden_weights = tf.get_variable(
                'U',
                [self.state_size, self.state_size])
            input_weights = tf.get_variable(
                'V',
                [self.input_size, self.state_size])
            a = tf.matmul(state, hidden_weights)
            b = tf.matmul(inputs, input_weights)
            bias = tf.get_variable('b', [self.state_size],
                                   initializer=tf.constant_initializer(0.0))
            output = tf.nn.relu(a * b + bias)
        return output, output
