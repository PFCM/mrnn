"""A more flexible vanilla/basic RNN cell. Possibly not as fast as the
tensorflow provided one, as it doesn't nicely roll everything up but
it allows for more flexible initialisation and nonlinearities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import mrnn.init as init


class VRNNCell(tf.nn.rnn_cell.RNNCell):
    """A basic rnn cell"""

    def __init__(self, num_units, input_size=None,
                 nonlinearity=tf.nn.tanh,
                 hh_init=tf.random_normal_initializer(stddev=0.15),
                 xh_init=tf.random_normal_initializer(stddev=0.15),
                 b_init=tf.constant_initializer(1.0, dtype=tf.float32)):
        """
        Sets up a cell.

        Args:
            num_units (int): the number of units. Equivalently the number of
                outputs and the size of the hidden state.
            input_size (int): the number of inputs.
            nonlinearity (callable): an elementwise nonlinearity to apply to
                the hidden state/outputs. Default is the classic tanh.
            hh_init (tensorflow initializer): an initialiser for the hidden
                to hidden weight matrix. Default is a centered normal with
                std dev of 0.15.
            xh_init (tensorflow initializer): an initialiser to the weight
                matrix applied to the input. Defaults to the same as `hh_init`.
            b_init (tensorflow initializer): an initialiser for the biases.
                Default is 1.0.
        """
        self._num_units = num_units
        self._input_size = input_size or num_units
        self._nonlinearity = nonlinearity
        self._hh_init = hh_init
        self._xh_init = xh_init
        self._b_init = b_init

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """output = new state = f(V*input + U*state + b)"""
        with tf.variable_scope(scope or type(self).__name__):
            input_weights = tf.get_variable('V',
                                            [self.input_size, self.state_size],
                                            initializer=self._xh_init)
            hidden_weights = tf.get_variable('W',
                                             [self.state_size, self.state_size],
                                             initializer=self._hh_init)
            bias = tf.get_variable('b', [self.state_size], initializer=self._b_init)

            a = tf.matmul(state, hidden_weights)
            b = tf.matmul(inputs, input_weights)
            output = self._nonlinearity(a + b + bias)
        return output, output


def IRNNCell(num_units, input_size=None, nonlinearity=tf.nn.relu):
    """Gets an IRNN cell as per http://arxiv.org/pdf/1504.00941.pdf
    although with no possibility of scaling the initialisation as
    of yet."""
    return VRNNCell(num_units, input_size,
                    nonlinearity=nonlinearity,
                    hh_init=init.identity_initializer(),
                    xh_init=init.identity_initializer(),
                    b_init=tf.constant_initializer(0., dtype=tf.float32))
