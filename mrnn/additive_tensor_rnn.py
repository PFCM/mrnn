"""An idea for a cell somewhat inspired by resnets/lstms
but incorporating a generalised two-way connection"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from mrnn.handy_ops import *
from mrnn.tensor_ops import *

import mrnn.init as init


class CPDeltaCell(tf.nn.rnn_cell.RNNCell):
    """Upon which all hopes are pinned"""

    def __init__(self, num_units, num_inputs, rank):
        self._num_units = num_units
        self._num_inputs = num_inputs
        self._rank = rank

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
        with tf.variable_scope(scope or type(self).__name__):
            input_weights = tf.get_variable('input_weights',
                                            [self.input_size,
                                             self.state_size])
            input_bias = tf.get_variable('input_bias', [self.state_size])
            input_adjustment = tf.nn.relu(
                tf.nn.bias_add(tf.matmul(inputs, input_weights), input_bias))

            with tf.variable_scope('tensor_product'):
                tensor = get_cp_tensor([self.state_size,
                                        self.output_size,
                                        self.state_size],
                                       self.rank,
                                       'weight_tensor',
                                       weightnorm=False,
                                       trainable=True)
                tensor_prod = bilinear_product_cp(input_adjustment,
                                                  tensor,
                                                  states)
            result = states + tensor_prod

        return result, result


class AddSubCPCell(tf.nn.rnn_cell.RNNCell):
    """Basically difference between two of the below"""

    def __init__(self, num_units, num_inputs, rank,
                 input_projections=None,
                 nonlinearity=tf.nn.relu,
                 tensor_init=init.spectral_normalised_init(0.999)):
        self._num_units = num_units
        self._num_inputs = num_inputs
        self._nonlinearity = nonlinearity
        self._rank = rank
        self._input_projection = input_projections or num_inputs
        self._tensor_init = tensor_init

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
        with tf.variable_scope(scope or type(self).__name__):
            # get the tensors first
            with tf.variable_scope('tensor_1',
                                   initializer=self._tensor_init):

                tensor_1 = get_cp_tensor([self.input_size,
                                          self.output_size,
                                          self.state_size],
                                         self.rank,
                                         'W1',
                                         weightnorm=False,
                                         trainable=True)
            with tf.variable_scope('tensor_2',
                                   initializer=self._tensor_init):
                tensor_2 = get_cp_tensor([self.input_size,
                                          self.output_size,
                                          self.state_size],
                                         self.rank,
                                         'W2',
                                         weightnorm=False,
                                         trainable=True)
            combo_1 = bilinear_product_cp(inputs, tensor_1, states)
            combo_2 = bilinear_product_cp(inputs, tensor_2, states)

            input_weights_1 = tf.get_variable('U1', shape=[self.input_size,
                                                           self._input_projection])
            input_weights_2 = tf.get_variable('U2', shape=[self.input_size,
                                                           self._input_projection])
            input_proj_1 = tf.matmul(inputs, input_weights_1)
            input_proj_2 = tf.matmul(inputs, input_weights_2)
            # biases
            bias1 = tf.get_variable('b1', shape=[self.output_size],
                                    initializer=tf.constant_initializer(0.0))
            bias2 = tf.get_variable('b2', shape=[self.output_size],
                                    initializer=tf.constant_initializer(0.0))
            positive = self._nonlinearity(combo_1 + input_proj_1 + bias1)
            negative = self._nonlinearity(combo_2 + input_proj_2 + bias2)
            result = positive - negative + states
        return result, result


class AdditiveCPCell(tf.nn.rnn_cell.RNNCell):
    """Uses a CP decomposition to factorise the tensor"""

    def __init__(self, num_units, num_inputs, rank,
                 input_projection=None,
                 nonlinearity=tf.nn.relu):
        self._num_units = num_units
        self._num_inputs = num_inputs
        self._nonlinearity = nonlinearity
        self._rank = rank
        self._input_projection = input_projection or num_inputs

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
        with tf.variable_scope(scope or type(self).__name__) as outer_scope:
            # do it
            # sub scope for the tensor init
            # should inherit reuse from outer scope
            with tf.variable_scope('tensor',
                                   initializer=init.spectral_normalised_init(1.1)):
                tensor = get_cp_tensor([self.input_size,
                                        self.output_size,
                                        self.state_size],
                                       self.rank,
                                       'W',
                                       weightnorm=False,
                                       trainable=True)
            combination = bilinear_product_cp(inputs, tensor, states)
            # and project the input
            input_weights = tf.get_variable('U', shape=[self.input_size,
                                                        self._input_projection],
                                            initializer=tf.uniform_unit_scaling_initializer(1.4))
            input_proj = tf.matmul(inputs, input_weights)
            # apply a bias pre-nonlinearity
            bias = tf.get_variable('b', shape=[self.output_size],
                                   initializer=tf.constant_initializer(0.0))
            result = self._nonlinearity(combination + input_proj + bias)
            result = result + states
        return result, result
