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

import mrnn.init as init


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
            # this is the transpose of the mode 1 unfolding
            # so it should be [statesize * statesize, inputs]
            tensor = random_sparse_tensor(
                [(self.state_size+1) * self.output_size, (self.input_size+1)],
                self.sparsity,
                name='W')
            # print('inputs: {}'.format(inputs.get_shape()))
            # I feel like this shouldn't happen but sometimes it does
            # if len(inputs.get_shape()) == 1:
            #     inputs = tf.expand_dims(inputs, 0)
            batch_ones = tf.ones([inputs.get_shape()[0].value, 1])
            vec_a = tf.concat(
                1, [inputs, batch_ones])
            vec_b = tf.concat(
                1, [states, batch_ones])
            activations = bilinear_product_sparse(vec_a, tensor, vec_b,
                                                  self.output_size,
                                                  batch_major=True)
            output = self._nonlinearity(activations)
            return output, output


class SimpleCPCell(tf.nn.rnn_cell.RNNCell):
    """Super simple net, but with a tensor stored in its
    CP approximation of given rank."""

    def __init__(self, num_units, num_inputs, rank,
                 nonlinearity=tf.nn.relu, weightnorm=False,
                 separate_pad=True):
        self._num_units = num_units
        self._num_inputs = num_inputs
        self._rank = rank
        self._nonlinearity = nonlinearity
        self._weightnorm = weightnorm
        self._separate_pad = separate_pad

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
        with tf.variable_scope(scope or type(self).__name__,
                               initializer=init.spectral_normalised_init(0.5)):
            # first we need to get the tensor
            if not self._separate_pad:
                shape = [self._num_units+1,
                         self._num_units,
                         self._num_inputs+1]

                vec_a = tf.concat(
                    1, [inputs, tf.ones([inputs.get_shape()[0].value, 1])])
                vec_b = tf.concat(
                    1, [states, tf.ones([inputs.get_shape()[0].value, 1])])
            else:
                shape = [self._num_units,
                         self._num_units,
                         self._num_inputs]
                vec_a, vec_b = inputs, states

            tensor = get_cp_tensor(shape,
                                   self._rank,
                                   'W',
                                   weightnorm=self._weightnorm)
            result = bilinear_product_cp(vec_a, tensor, vec_b)

            if self._separate_pad:
                # TODO: inits
                # should we roll these up into one matmul?
                # probably will be faster
                if self._weightnorm:
                    in_weights = get_weightnormed_matrix(
                        [self._num_inputs, self._num_units],
                        name='input_weights')
                    rec_weights = get_weightnormed_matrix(
                        [self._num_units, self._num_units],
                        name='recurrent_weights',
                        V_init=init.identity_initializer())
                else:
                    in_weights = tf.get_variable(
                        'input_weights',
                        [self._num_inputs, self._num_units],
                        tf.float32,
                        initializer=tf.uniform_unit_scaling_initializer())
                    rec_weights = tf.get_variable(
                        'recurrent_weights',
                        [self._num_units, self._num_units],
                        tf.float32,
                        initializer=init.identity_initializer())
                bias = tf.get_variable('bias',
                                       [self._num_units],
                                       initializer=tf.constant_initializer(0.0))
                result += tf.nn.bias_add(
                    tf.matmul(vec_a, in_weights) + tf.matmul(vec_b, rec_weights),
                    bias)

            result = self._nonlinearity(result)
            return result, result


class SimpleTTCell(tf.nn.rnn_cell.RNNCell):
    """Simple RNN cell using the tensor stored in TT format"""

    def __init__(self, num_outputs, num_inputs, ranks,
                 nonlinearity=tf.nn.tanh, separate_pad=True):
        """make the thing.

        Args:
          num_outputs: how many outputs this layer should have. Also
            the size of the hidden state.
          num_inputs: how many inputs this layer has.
          ranks: a sequence of two integers defining the TT ranks of
            the approximation.
          nonlinearity: some kind of function we can apply elementwise
            to the output. This just happens to the output, we don't
            use this class to experiment with pushing it around.
          separate_pad: if False, we add ones to the input and state
            and make the weights bigger. If True, we pull these components
            out and treat them separately, which may aid learning.
        """
        self._num_outputs = num_outputs
        self._num_inputs = num_inputs
        self._nonlin = nonlinearity
        self._separate_pad = separate_pad
        if len(ranks) != 2:
            raise ValueError('Need two ranks, got: {}'.format(ranks))
        self._ranks = ranks

    @property
    def ranks(self):
        return self._ranks

    @property
    def state_size(self):
        return self._num_outputs

    @property
    def input_size(self):
        return self._num_inputs

    @property
    def output_size(self):
        return self._num_outputs

    def __call__(self, inputs, states, scope=None):
        with tf.variable_scope(
                scope or type(self).__name__,
                initializer=tf.random_normal_initializer(stddev=0.01)):
            # get the tensor
            if self._separate_pad:
                t_shape = [self._num_outputs,
                           self._num_outputs,
                           self._num_inputs]
                vec_a = inputs
                vec_b = states
            else:
                t_shape = [self._num_outputs+1,
                           self._num_outputs,
                           self._num_inputs+1]
                vec_a = tf.concat(
                    1, [inputs, tf.ones([inputs.get_shape()[0].value, 1])])
                vec_b = tf.concat(
                    1, [inputs, tf.ones([inputs.get_shape()[0].value, 1])])
            tensor = get_tt_3_tensor(t_shape, self._ranks, name='W')
            result = bilinear_product_tt_3(vec_a, tensor, vec_b)
            if self._separate_pad:
                # TODO possible weightnorm
                D = tf.get_variable('D', [self._num_inputs, self._num_outputs],
                                    initializer=tf.uniform_unit_scaling_initializer(1.2))
                E = tf.get_variable('E', [self._num_outputs, self._num_outputs],
                                    initializer=tf.uniform_unit_scaling_initializer(1.2))
                b = tf.get_variable('b', [self._num_outputs],
                                    initializer=tf.constant_initializer(0.0))
                z = tf.nn.bias_add(tf.matmul(inputs, D) + tf.matmul(states, E), b)
                result = result + z

            result = self._nonlin(result)
            return result, result
