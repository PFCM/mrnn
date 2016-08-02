"""A more flexible vanilla/basic RNN cell. Possibly not as fast as the
tensorflow provided one, as it doesn't nicely roll everything up but
it allows for more flexible initialisation and nonlinearities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import mrnn.init as init
import mrnn.handy_ops as hops


class FlatRNNCell(tf.nn.rnn_cell.RNNCell):
    """a basic rnn cell, but rolled up to use one big weight
    matrix, as per tf.nn.rnn_cell.BasicRNNCell except with more
    flexible nonlinearities etc."""

    def __init__(self, num_units, input_size=None,
                 nonlinearity=tf.nn.tanh,
                 W_init=tf.random_normal_initializer(stddev=0.15),
                 b_init=tf.constant_initializer(0.0, dtype=tf.float32),
                 weightnorm=False):
        """set up the cell.

        Args:
          num_units (int): how many cells/outputs.
          input_size (int): how many inputs.
          W_init (tensorflow initializer): initialiser for the big weight
            matrix, default is random normal with mean 0 and stddev 0.15.
          weightnorm: whether to weight normalise the rows of the resulting
            matrix.
        """
        self._num_units = num_units
        self._input_size = input_size or num_units
        self._W_init = W_init
        self._b_init = b_init
        self._weightnorm = weightnorm
        self._nonlin = nonlinearity

    @property
    def input_size(self):
        return self._input_size

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """output = new state = f(W * (input; scope) + b)"""
        with tf.variable_scope(scope or type(self).__name__):
            if self._weightnorm:
                weights = hops.get_weightnormed_matrix(
                    [self.state_size,
                     self.input_size + self.state_size],
                    name='W', V_init=self.W_init)
            else:
                weights = tf.get_variable('W',
                                          [self.input_size + self.state_size,
                                           self.state_size],
                                          initializer=self._W_init)
            bias = tf.get_variable('b', [self.state_size])
            args = tf.concat(1, [inputs, state])
            output = self._nonlin(tf.matmul(args, weights) + bias)
        return output, output


class VRNNCell(tf.nn.rnn_cell.RNNCell):
    """A basic rnn cell"""

    def __init__(self, num_units, input_size=None,
                 nonlinearity=tf.nn.tanh,
                 hh_init=tf.random_normal_initializer(stddev=0.15),
                 xh_init=tf.random_normal_initializer(stddev=0.15),
                 b_init=tf.constant_initializer(0.0, dtype=tf.float32),
                 weight_noise=0.0,
                 keep_prob=1.0,
                 weightnorm=False):
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
            weightnorm: whether or not to use weight normalisation on the
                various weight matrices. Values are: 'none', 'full',
                'recurrent' or 'input'
        """
        self._num_units = num_units
        self._input_size = input_size or num_units
        self._nonlinearity = nonlinearity
        self._hh_init = hh_init
        self._xh_init = xh_init
        self._b_init = b_init
        self._weightnorm = weightnorm
        self._keep_prob = keep_prob
        self._weight_noise = weight_noise

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
            if self._weightnorm == 'full' or self._weightnorm == 'input':
                input_weights = hops.get_weightnormed_matrix(
                    [self.input_size, self.state_size],
                    name='V', V_init=self._xh_init)

            else:
                input_weights = tf.get_variable(
                    'V',
                    [self.input_size, self.state_size],
                    initializer=self._xh_init)
            if self._weightnorm == 'full' or self._weightnorm == 'recurrent':
                hidden_weights = hops.get_weightnormed_matrix(
                    [self.state_size, self.state_size],
                    name='W', V_init=self._hh_init)
            else:
                hidden_weights = tf.get_variable(
                    'W',
                    [self.state_size, self.state_size],
                    initializer=self._hh_init)
            bias = tf.get_variable('b', [self.state_size],
                                   initializer=self._b_init)

            if self._weight_noise != 0.0:
                hidden_weights = hops.variational_wrapper(
                    hidden_weights, weight_noise=self._weight_noise,
                    name='hidden_weightnoise')
                input_weights = hops.variational_wrapper(
                    input_weights, weight_noise=self._weight_noise,
                    name='input_weightnoise')
            if self._keep_prob != 0.0:
                inputs = hops.variational_wrapper(
                    inputs, keep_prob=self._keep_prob,
                    name='input_dropout')
                state = hops.variational_wrapper(
                    state, keep_prob=self._keep_prob,
                    name='state_dropout')

            a = tf.matmul(state, hidden_weights)
            b = tf.matmul(inputs, input_weights)
            pre_activations = a + b
            if self._weightnorm == 'layer':
                pre_activations = hops.layer_normalise(pre_activations)
            output = self._nonlinearity(pre_activations + bias)
        return output, output


def IRNNCell(num_units, input_size=None, nonlinearity=tf.nn.relu,
             weightnorm='none'):
    """Gets an IRNN cell as per http://arxiv.org/pdf/1504.00941.pdf
    although with no possibility of scaling the initialisation as
    of yet."""
    return VRNNCell(num_units, input_size,
                    nonlinearity=nonlinearity,
                    hh_init=init.identity_initializer(),
                    # xh_init=init.identity_initializer(),
                    b_init=tf.constant_initializer(0., dtype=tf.float32),
                    weightnorm=weightnorm)
