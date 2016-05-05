"""Some handy tensor guys. In particular for computing bilinear
products given various different representations of the central
tensor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

import mrnn.handy_ops


def get_cp_tensor(shape, maxrank, name, weightnorm=False, dtype=tf.float32):
    """Gets the components of a tensor stored in its CP decomposition.
    Rather than `prod(shape)` elements, this form will have
    `sum(maxrank * prod)`.

    Concretely, if `shape = [100, 100, 100]` then explicitly storing
    the tensor would take 1,000,000 numbers. If we use an approximation
    of rank 100, the number of parameters is `100*100*3 = 30,000`.


    Note that this does not use a variable_scope, only a name_scope,
    so it should play nice with reusing variables. The initialisation
    is also the default initialiser for whatever variable scope this
    function is called in.

    Args:
      shape: the shape of what the tensor would be were it stored
        explicitly.
      maxrank: the number of rank-1 tensors whose tensor products
        we sum over. This is then the maximum rank of the tensor and
        controls the number of parameters as well as some notion of
        the expressive power of the tensor.
      name: the name, this is used to build the scope under which the
        variables are allocated.
      weightnorm: whether or not to use weight normalisation on the
        returned matrices. If so then maybe it will learn better?
      dtype: the data type of the resulting matrices (defaults to
        tf.float32)

    Returns:
      tuple of `len(shape)` matrices, with each one of shape `[maxrank x d]`
        where `d` is the corresponding entry in `shape`
    """
    with tf.name_scope(name):
        matrices = []
        for i, dim in enumerate(shape):
            matrices.append(
                tf.get_variable('cp_decomp_{}'.format(i),
                                [maxrank, dim],
                                dtype=dtype))
    return tuple(matrices)


def bilinear_product_cp(vec_a, tensor, vec_b, batch_major=True,
                        name='bilinear_cp'):
    """Does the ops to do a bilinear product of the form:
    $ a^TWb $ where $a$ and $b$ are vectors and $W$ a 3-tensor stored as
    a tuple of three matrices (as per `get_cp_tensor`).

    Should be done in such a way that vec_a and vec_b can be either vectors
    or matrices containing a batch of data.

    Args:
      vec_a: the vector on the left hand side (although the order shouldn't
        matter).
      tensor: the tensor in the middle. Expected to in fact be a sequence of
        3 matrices. We assume these are ordered such that if vec_a is shape
        [a,] and vec_b is shape [b,] then tensor[0] is shape [rank, a],
        tensor[1] is shape [rank, x] and tensor[2] is shape [rank, b]. The
        result will be [x,]
      vec_b: the remaining vector.
      batch_major: if true, we expect the data (vec_a and vec_b) to be of
        shape `[batch_size, -1]`, otherwise the opposite.
      name: a name for the ops

    Returns:
      the result.

    Raises:
      ValueError: if the various shapes etc don't line up.
    """
    # quick sanity checks
    if len(tensor) != 3:
        raise ValueError('Expecting three way decomposed tensor')

    with tf.name_scope(name):
        # TODO(pfcm) performance evaluation between concatenating these or not
        # (probably will be faster, but maybe not if we have to do it every
        # time)
        # alternative:
        # prod_a_b = tf.matmul(
        #     tf.concatenate(1, (tensor[0], tensor[2])),
        #     tf.concatenate(0, (vec_a, vec_b)))
        prod_a = tf.matmul(tensor[0], vec_a, transpose_b=batch_major)
        prod_c = tf.matmul(tensor[2], vec_b, transpose_b=batch_major)
        # now do these two elementwise
        prod_b = tf.mul(prod_a, prod_c)
        # and multiply the result by the remaining matrix in tensor
        result = tf.matmul(tensor[1], prod_b, transpose_a=True)
        if batch_major:
            result = tf.transpose(result)
    return result
