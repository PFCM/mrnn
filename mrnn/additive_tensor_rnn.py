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


def _tensor_logits(inputs, states, rank, weightnorm=None, pad=True,
                   separate_pad=True, name='t_prod'):
    """Gets a tensor product of inputs and states, with a few options.

    Args:
        inputs: the inputs to the layer (or just the first input).
        states: the previous hidden states (or just the second input).
        rank: the rank of the decomposed tensor to use.
        weightnorm: Whether or not (or indeed how) to normalise the weights.
            Possible values are `None` for a totally unconstrained tensor,
            'partial' for doing a minimal normalisation (with no extra
            parameters)  which just returns each of the matrices divided by
            its squared frobenius norm or 'classic' which does all of the
            rows of all of the matrices and adds the necessary learnable
            gain coefficients.
        pad: whether or not to add biases to the tensor product.
        separate_pad: whether or not to incorporate the bias matrices into
            the decomposition. If true, we will have more parameters but we
            will definitely be capable of exactly representing a classic RNN.

    Returns:
        tensor, the result.
    """
    batch_size, state_size = states.get_shape().as_list()
    input_size = inputs.get_shape()[1].value
    if pad and not separate_pad:
        # just add a column of ones to inputs and states
        inputs = tf.concat(1, [tf.ones([batch_size, 1]), inputs])
        states = tf.concat(1, [tf.ones([batch_size, 1]), states])
        # the tensor now has to be a little bit of an odd shape
        tensor = get_cp_tensor([state_size+1,
                                state_size,
                                input_size+1],
                               rank,
                               name,
                               weightnorm=weightnorm)
    else:
        # the tensor is normal shape
        tensor = get_cp_tensor([state_size,
                                state_size,
                                input_size],
                               rank,
                               name,
                               weightnorm=weightnorm)
    tensor_prod = bilinear_product_cp(states, tensor, inputs)

    if pad and separate_pad:
        # then we have to do these guys too
        input_weights = possibly_weightnormed_var([input_size, state_size],
                                                  weightnorm,
                                                  name + 'input_weights')
        state_weights = possibly_weightnormed_var([state_size, state_size],
                                                  weightnorm,
                                                  name + 'state_weights')
        bias = tf.get_variable(name+'bias', dtype=tf.float32,
                               shape=[state_size],
                               initializer=tf.constant_initializer(0.0))
        tensor_prod += tf.nn.bias_add(
            tf.matmul(inputs, input_weights) +
            tf.matmul(states, state_weights),
            bias)
    return tensor_prod


def _affine(data, new_size, weightnorm=None, name='affine'):
    """performs an affine transormation on the data using trainable, possibly weightnormed
    variables"""
    input_size = data.get_shape()[1].value
    weights = possibly_weightnormed_var([input_size, new_size],
                                        weightnorm,
                                        name+'_weights')
    bias = tf.get_variable(name+'_bias', dtype=tf.float32, shape=[new_size],
                           initializer=tf.constant_initializer(0.0))
    return tf.nn.bias_add(tf.matmul(data, weights), bias)


class CPGateCell(tf.nn.rnn_cell.RNNCell):
    """A forget gate and an accumulator"""

    def __init__(self, num_units, rank, dropout=1.0, layernorm='',
                 weight_noise=0.0):
        self._num_units = num_units
        self._rank = rank
        self._dropout = dropout
        self.layernorm = layernorm
        self.weight_noise = weight_noise

    @property
    def rank(self):
        return self._rank

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope('accumulator',
                                   initializer=init.orthonormal_init(1.0)):
                input_acts = _affine(inputs, self.state_size)
                if 'pre' in self.layernorm and 'input' in self.layernorm:
                    input_acts = layer_normalise(input_acts)
                update = tf.nn.relu(input_acts)
                if self._dropout != 1.0:
                    # update = tf.nn.dropout(update, self._dropout)
                    update = variational_wrapper(
                        update, keep_prob=self._dropout)

            with tf.variable_scope('forgetter',
                                   initializer=init.orthonormal_init(1.0)):
                forget_acts = _tensor_logits(
                    inputs, state, self.rank, pad=True, separate_pad=True)
                if 'pre' in self.layernorm and 'forget' in self.layernorm:
                    forget_acts = layer_normalise(forget_acts, add_bias=True,
                                                  bias_initialiser=0.0)
                forget_gate = tf.nn.sigmoid(forget_acts)

            result = forget_gate * state + (1.0-forget_gate) * update
        return result, result


class CPResCell(tf.nn.rnn_cell.RNNCell):
    """try something JIC"""

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
            # first combine the inputs and the state
            with tf.variable_scope('pre_act', initializer=init.orthonormal_init(0.5)):
                pre_activations = _tensor_logits(inputs, states, self.rank,
                                                 weightnorm=None,
                                                 pad=False,
                                                 separate_pad=False,
                                                 name='pre_act')
                # layer norm with extra bias
                pre_activations = layer_normalise(pre_activations, add_bias=True)
            # make the cut
            cut = tf.nn.relu(pre_activations)
            # project back into weight pace
            with tf.variable_scope('post_act', initializer=init.orthonormal_init(0.5)):
                post_activations = _affine(pre_activations, self.state_size, name='cut_proj',
                                           weightnorm=None)
                # re-normalise
                result = layer_normalise(post_activations)
            # make the skip connection
            result += states
        return result, result



class CPDeltaCell(tf.nn.rnn_cell.RNNCell):
    """Upon which all hopes are pinned"""

    def __init__(self, num_units, num_inputs, rank, weightnorm=None):
        self._num_units = num_units
        self._num_inputs = num_inputs
        self._rank = rank
        self.weightnorm = weightnorm

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
            with tf.variable_scope('plus_tensor',
                                   initializer=init.spectral_normalised_init(1.1)):
                pos_tensor_prod = _tensor_logits(inputs, states, self.rank,
                                                 weightnorm=self.weightnorm,
                                                 pad=True,
                                                 separate_pad=False,
                                                 name='positive')

                positive = tf.nn.relu(pos_tensor_prod)

            with tf.variable_scope('minus_tensor',
                                   initializer=init.spectral_normalised_init(1.1)):
                neg_tensor_prod = _tensor_logits(inputs, states, self.rank,
                                                 weightnorm=self.weightnorm,
                                                 pad=True,
                                                 separate_pad=False,
                                                 name='negative')
                negative = tf.nn.relu(neg_tensor_prod)

            result = positive - negative + states
        return result, result



class CPLossyIntegrator(tf.nn.rnn_cell.RNNCell):
    """Upon which all hopes are pinned"""

    def __init__(self, num_units, num_inputs, rank, weightnorm=None):
        self._num_units = num_units
        self._num_inputs = num_inputs
        self._rank = rank
        self.weightnorm = weightnorm

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
            # project the input into the hidden feature space
            with tf.variable_scope('input_projection', initializer=tf.random_normal_initializer(stddev=0.01)):
                proj_in = _affine(inputs, self.state_size, name='cut_shift')
                proj_in = tf.nn.relu(proj_in)
                proj_in = _affine(inputs, self.state_size, name='state_proj')
            # compute via a bilinear product some parts of the memory to abrade
            with tf.variable_scope('abrasion', initializer=init.spectral_normalised_init(1.5)):
                forgetting = _tensor_logits(inputs, states, self.rank,
                                            pad=True, separate_pad=False)
                forgetting = tf.nn.relu(forgetting)
                forgetting = _affine(forgetting, self.state_size)

            # combine the information
            result = tf.nn.relu(states + proj_in - forgetting)

        return result, result


class CPSimpleIntegrator(tf.nn.rnn_cell.RNNCell):
    """Upon which all hopes are pinned"""

    def __init__(self, num_units, num_inputs, rank, layernorm=False):
        self._num_units = num_units
        self._num_inputs = num_inputs
        self._rank = rank
        self.layernorm = layernorm

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
            # project the input into the hidden feature space
            with tf.variable_scope('input_projection'):
                pre_acts = _affine(inputs, self.state_size)
                if self.layernorm == 'pre':
                    pre_acts = layer_normalise(pre_acts)
                input_info = tf.nn.relu(pre_acts)
            # compute via a bilinear product some parts of the memory to abrade
            with tf.variable_scope('abrasion'):
                neg_acts = _tensor_logits(inputs, states, self.rank, pad=False)
                if self.layernorm == 'pre':
                    neg_acts = layer_normalise(neg_acts, add_bias=True, bias_init=1.0)
                combined = -tf.nn.relu(neg_acts)

            if self.layernorm == 'post':
                update = layer_normalise(input_info + combined)
            else:
                update = input_info + combined
        result = tf.nn.relu(states + update)
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
                                         weightnorm='partial',
                                         trainable=True)
            with tf.variable_scope('tensor_2',
                                   initializer=self._tensor_init):
                tensor_2 = get_cp_tensor([self.input_size,
                                          self.output_size,
                                          self.state_size],
                                         self.rank,
                                         'W2',
                                         weightnorm='partial',
                                         trainable=True)
            combo_1 = bilinear_product_cp(inputs, tensor_1, states)
            combo_2 = bilinear_product_cp(inputs, tensor_2, states)

            input_weights_1 = tf.get_variable('U1', shape=[self.input_size,
                                                           self.state_size])
            input_weights_2 = tf.get_variable('U2', shape=[self.input_size,
                                                           self.state_size])
            input_proj_1 = tf.matmul(inputs, input_weights_1)
            input_proj_2 = tf.matmul(inputs, input_weights_2)
            # biases
            bias1 = tf.get_variable('b1', shape=[self.output_size],
                                    initializer=tf.constant_initializer(0.0))
            bias2 = tf.get_variable('b2', shape=[self.output_size],
                                    initializer=tf.constant_initializer(0.0))
            positive = self._nonlinearity(combo_1 + input_proj_1 + bias1)
            # positive = tf.nn.l2_normalize(positive, 1)
            negative = self._nonlinearity(combo_2 + input_proj_2 + bias2)
            # negative = tf.nn.l2_normalize(negative, 1)
            result = positive - negative + states
        return result, result


class AdditiveCPCell(tf.nn.rnn_cell.RNNCell):
    """Uses a CP decomposition to factorise the tensor"""

    def __init__(self, num_units, num_inputs, rank,
                 input_projection=None,
                 nonlinearity=tf.nn.relu, layernorm=False):
        self._num_units = num_units
        self._num_inputs = num_inputs
        self._nonlinearity = nonlinearity
        self._rank = rank
        self._input_projection = input_projection or num_units
        self.layernorm = layernorm

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
                                   initializer=init.orthonormal_init(0.5)):
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
            if self.layernorm == 'pre':
                activations = layer_normalise(combination + input_proj + bias)
            else:
                activations = combination + input_proj + bias

            result = self._nonlinearity(activations)

            if self.layernorm == 'post':
                result = layer_normalise(result)

            result = result + states
        return result, result
