"""Some convenience stuff"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def possibly_weightnormed_var(shape, normtype, name, trainable=True):
    """Gets a variable which may or may not be weightnormed, depending on the
    `normtype`."""
    if normtype == 'classic':
        # column-wise l2 norms, with learnable gains
        return get_weightnormed_matrix(shape, name=name, V_init=None,
                                       trainable=trainable)
    elif normtype == 'partial':
        # frobenius, no gains
        return get_weightnormed_matrix(shape, axis=None, name=name,
                                       V_init=None, squared=False,
                                       trainable=trainable)
    elif normtype == 'row':
        # do it by row, but still no gains
        return get_weightnormed_matrix(shape, axis=0, name=name,
                                       V_init=None, train_gains=False)
    elif not normtype:
        # unconstrained
        return tf.get_variable(name, dtype=tf.float32, shape=shape,
                               trainable=trainable)
    else:
        raise ValueError('Unknown weightnorm type: {}'.format(normtype))


def get_weightnormed_matrix(shape, axis=1, name='weightnorm',
                            V_init=tf.random_normal_initializer(stddev=0.015),
                            train_gains=True, dtype=tf.float32,
                            trainable=True, squared=False):
    """Returns a matrix weightnormed across a given index.

    Adds 2 trainable variables:
      - V, a matrix, initialised with the default init
      - g, a vector, initialised to 1s

    returns g * V / elementwise l2 norm of V.

    Args:
        shape: sequence of 2 ints. We are only dealing with matrices
            here.
        axis: how to do the normalising, defaults to 1, which is likely
            to be what you want if your data is `[batch_size x d]`.
        name: name for the scope, defaults to weightnorm
        V_init: initialiser for the unnormalised part of the matrix.
        train_gains: if false, gains will be always one.
        dtype: type for the created variables.
        trainable: whether the matrix should be added to the tensorflow
            trainable variables collection.
        squared: if true, don't take the square root and just divide by the
            squared norm.

    Returns:
        Tensor: the matrix whose rows or columns will never exceed the learned
            norm.
    """
    if len(shape) != 2:
        raise ValueError(
            'Expected two dimensional shape, but it is {}'.format(shape))
    with tf.name_scope(name):
        unnormed_w = tf.get_variable(name+'_V', shape,
                                     trainable=trainable,
                                     initializer=V_init,
                                     dtype=dtype)
        if axis:
            gains = tf.get_variable(name+'_g', [shape[0], 1],
                                    trainable=train_gains,
                                    initializer=tf.constant_initializer(1.0),
                                    dtype=dtype)
        else:
            gains = 1.0
        sqr_norms = tf.reduce_sum(
                tf.square(unnormed_w),
                reduction_indices=axis,
                keep_dims=True)

        if not squared:
            inv_norms = tf.rsqrt(sqr_norms)
        else:
            inv_norms = 1.0 / sqr_norms

        return gains * unnormed_w * inv_norms


def layer_normalise(activations, gain_initialiser=1.0, bias_initialiser=0.0,
                    add_bias=False):
    """Performs layer normalisation, as per 
    https://arxiv.org/abs/1607.06450

    This adds a few variables and will attempt to reuse them if the outer
    variable scope desires this.

    Args:
        activations: the computed activations for a layer that need to be re-scaled.
            this is assumed to be `[batch_size, features]`.
        gain_initialiser: where to start the gain parameters.
        bias_initialiser: where to start the extra biases.
        add_bias: whether to actually add biases -- there is a good chance you are
            already adding biases when computing the activations, so it might be
            pointless to add more.

    Returns:
        layer normalised activations -- see the paper for the precise formulation.
    """
    with tf.variable_scope('layer_normalisation'):
        # first we need to compute the statistics
        # this is done per example (we don't reduce through the batch dimension)
        mu = tf.reduce_mean(activations, reduction_indices=1,
                            keep_dims=True)  # [batch_size, 1]
        # now center
        centered = activations - mu  # need this to broadcast
        
        sigma = tf.reduce_mean(tf.square(centered),  # hope for broadcast
                               reduction_indices=1, keep_dims=True)
        inv_sigma = tf.rsqrt(sigma)
        # get the gain coefficients
        gains = tf.get_variable('gains', shape=[1, activations.get_shape()[1].value],
                                initializer=tf.constant_initializer(gain_initialiser))
        # getting these dimensions to work out is a little bit awkward
        # the easiest way (but probably not the best) is to take the
        # outer product of the gains and the variances giving us a matrix
        # where each row is the gain vector multiplied by the inverse variance for the
        # for that case. Storing this explicitly is probably wasteful.
        scale = tf.matmul(inv_sigma, gains)
        # elementwise multiply
        result = scale * centered
        # possibly add bias
        if add_bias:
            bias = tf.get_variable('biases', shape=[activations.get_shape()[1].value],
                                   initializer=tf.constant_initializer(bias_initialiser))
            result = tf.nn.bias_add(result, bias)
        return result
