"""Some convenience stuff"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def get_weightnormed_matrix(shape, axis=1, name='weightnorm',
                            V_init=tf.random_normal_initializer(stddev=0.015),
                            train_gains=True, dtype=tf.float32):
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

    Returns:
        Tensor: the matrix whose rows or columns will never exceed the learned norm.
    """
    if len(shape) != 2:
        raise ValueError('Expected two dimensional shape, but it is {}'.format(shape))
    with tf.name_scope(name):
        unnormed_w = tf.get_variable(name+'_V', shape, trainable=True,
                                     initializer=V_init,
                                     dtype=dtype)
        gains = tf.get_variable(name+'_g', [shape[0], 1], trainable=train_gains,
                                initializer=tf.constant_initializer(1.0),
                                dtype=dtype)
        inv_norms = tf.rsqrt(
            tf.reduce_sum(
                tf.square(unnormed_w),
                reduction_indices=1,
                keep_dims=True))
        return gains * unnormed_w * inv_norms
