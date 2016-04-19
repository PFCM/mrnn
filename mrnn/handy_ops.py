"""Some convenience stuff"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf


def flat_3tensor_product(t_a, m_b, output_shape,
                             name='flat_tensor_product',
                             sparse_tensor=True):
    """Computes the product of a flattened three-way tensor and
    a matrix, reshaping the output appropriately to the final matrix.

    Args:
        t_a: the (optionally sparse) flattened 3-tensor. The
            way in which it is matricised determines wihch of its
            indices gets squashed (ie. a mode-3 matricisation would
            result in this being the mode-3 product).
        m_b: the the dense matrix/vector we are getting a result with.
        output_shape: the resulting size to reshape the output to.
        name: a name for any added ops.
        sparse_tensor: if `tensor` is an instance of tf.SparseTensor
            then we can (and very much want to) use more efficient 
            matmul.

    Returns:
        dense matrix with shape `output_shape`

    Raises:
        probably a lot, does no checking to make sure it has sane
        inputs.
    """
    with tf.name_scope(name):
        if sparse_tensor:
            # see
            # https://www.tensorflow.org/versions/r0.8/api_docs/python/sparse_ops.html#sparse_tensor_dense_matmul
            # and make sure the t_a is appropriately sorted and the
            # right way around so the multiply works out
            result = tf.sparse_tensor_dense_matmul(t_a, m_b)
        else:
            # standard matmul, same caveats apply re the shape
            result = tf.matmul(t_a, m_b)
        return tf.reshape(result, output_shape)


def random_sparse_tensor(shape, sparsity, stddev=0.01, name='random-sparse'):
    """Returns a sparse tensor with a set sparsity but
    with random indices and values.

    Values are from a normal with mean 0 and given std deviation.

    Args:
        shape: list of ints, the final shape of the
            tensor.
        sparsity: scalar float tensor of float, the sparsity,
            0 < sparsity < 1.
    """
    if type(sparsity) != tf.Tensor:
        if sparsity <= 0 or sparsity >=1:
            raise ValueError('sparsity {} is out of range (0-1)'.format(sparsity))
    size = 1
    for dim in shape:
        size *= dim
    # now how many non-zero
    num_elements = int(size * sparsity)
    logging.info('%d elements (out of %d, shape %s)', num_elements, size, shape)
    # the first thing we need are random indices
    idces = tf.pack([tf.get_variable(name+'-index-{}'.format(i),
                                     initializer=tf.random_uniform(
                                         [num_elements],
                                         minval=0,
                                         maxval=dim,
                                         dtype=tf.int64),
                                     dtype=tf.int64)
                     for i, dim in enumerate(shape)])
    idces = tf.transpose(idces)
    # we should check for repeats
    print('you should check for repeats in the sparse tensor indices')
    # and now values
    vals = tf.get_variable(name+'values',
                           [num_elements],
                           initializer=tf.random_normal_initializer(
                               stddev=stddev))
    return tf.SparseTensor(idces, vals, shape)
