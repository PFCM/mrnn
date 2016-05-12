"""Some handy tensor guys. In particular for computing bilinear
products given various different representations of the central
tensor."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import tensorflow as tf

import mrnn.handy_ops as hops

logger = logging.getLogger(__name__)


def get_sparse_tensor(shape, sparsity, stddev=0.15, name='random-sparse'):
    """wrapper for more genereal `random_sparse_tensor` function.
    Accepts a 3D shape that, returns an appropriately unfolded tensor to
    use with `bilinear_product_sparse`.

    Args:
        shape: 3 ints, shape of tensor.
        sparsity: see `random_sparse_tensor`
        stddev: for initialisation
        name: name of ops etc.

    Returns:
        the transpose of the mode one unfolding of a sparse tensor.
    """
    assert len(shape) == 3
    return random_sparse_tensor([shape[1]*shape[2], shape[0]],
                                sparsity, stddev=stddev, name=name)


def random_sparse_tensor(shape, sparsity, stddev=0.15, name='random-sparse'):
    """Returns a sparse tensor with a set sparsity but
    with random indices and values.

    Values are from a normal with mean 0 and given std deviation.

    Args:
        shape: list of ints, the final shape of the
            tensor.
        sparsity: scalar, If it is an integer > 0 it is assumed to be the
            number of elements in the sparse tensor, otherwise if it is a
            float in [0, 1] it is treated as a fraction of elements to set.
        stddev: the standard deviation of the values.
        name: the name of the tensor
    """
    if int(sparsity) != 0:
        logger.info('assuming sparsity is number of elements')
        num_elements = int(sparsity)
    elif sparsity <= 0 or sparsity >= 1:
        raise ValueError(
            'sparsity {} is out of range (0-1)'.format(sparsity))
    else:
        logger.info('assuming sparsity (%.3f) is fraction', sparsity)

        size = 1
        for dim in shape:
            size *= dim
        # now how many non-zero
        num_elements = int(size * sparsity)
        logger.info('(amounts to %d elements)', num_elements)
    # the first thing we need are random indices
    # it's a bit hard to do this without the possibility of repeats
    idces = tf.pack([tf.cast(
        tf.get_variable(name+'_idcs{}'.format(i),
                        shape=[num_elements],
                        initializer=tf.random_uniform_initializer(
                            maxval=dim,
                            dtype=tf.float32),
                        dtype=tf.float32),
        tf.int64)
                     for i, dim in enumerate(shape)])
    idces = tf.transpose(idces)  # should check for repeats?
    # and now values
    vals = tf.get_variable(name+'values',
                           [num_elements],
                           initializer=tf.random_normal_initializer(
                               stddev=stddev))
    return tf.sparse_reorder(tf.SparseTensor(idces, vals, shape))


def bilinear_product_sparse(vec_a, tensor, vec_b, output_size,
                            batch_major=True):
    """Performs a bilinear product with a sparse tensor.
    If vec_a is [I x 1], vec_b is [K x 1], tensor should be the transpose of a
    mode one unfolding of a [I x J x K] tensor (so a [JK x I] sparse matrix).

    Args:
        vec_a: a vector
        tensor: transpose of mode one unfolding of sparse tensor
        vec_b: a vector
        output_size: the length of the output vectors. Turns out to be handy.
        batch_major: whether the vectors are actually [batch_size x {I,J}]

    Returns:
        a vector, with the length of the middle dimension of tensor when it is
            not unfolded.
    """
    # turns out we want vec_a to be [I x B]
    # and vec_b t be [B x K]
    if batch_major:
        vec_a = tf.transpose(vec_a)
    else:
        vec_b = tf.transpose(vec_b)
    # we can just to this as a matmul, a reshape and a batch matmul
    temp = tf.sparse_tensor_dense_matmul(tensor, vec_a)
    # result will be [JK x batch_size]
    temp = tf.transpose(temp)  # make sure the reshaping is reshaping properly
    temp = tf.reshape(temp, [vec_b.get_shape()[0].value,
                             output_size,
                             vec_b.get_shape()[1].value])
    # now have temp = [B x J x K]
    # we need to add a trailing 1 to the shape of vec_b so it is [B x K x 1]
    # and we squeeze the result so it should be back to 2D [B x J]
    # print(temp.get_shape())
    # print(tf.expand_dims(vec_b, 2).get_shape())
    return tf.squeeze(tf.batch_matmul(temp, tf.expand_dims(vec_b, 2)), [2])


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
            if weightnorm:
                matrices.append(
                    hops.get_weightnormed_matrix(
                        [maxrank, dim],
                        name='cp_decomp_{}'.format(i)))
            else:
                matrices.append(
                    tf.get_variable('cp_decomp_{}'.format(i),
                                    [maxrank, dim],
                                    dtype=dtype, trainable=True))
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


#def get_tt_tensor(size, 
