"""Contains some initialisers which it has proved prudent to
implement"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf



def identity_initializer():
    """Initialise a variable to the identity.
    Due to https://github.com/tensorflow/tensorflow/issues/434
    with minor modifications.
    """
    def _initializer(shape, dtype=tf.float32):
        if len(shape) == 1:
            return tf.constant(1., shape=shape, dtype=dtype)
        elif len(shape) == 2 and shape[0] == shape[1]:
            return tf.constant(np.identity(shape[0]), dtype=dtype)
        elif len(shape) == 4 and shape[2] == shape[3]:
            array = np.zeros(shape)
            cx, cy = shape[0]/2, shape[1]/2
            for i in range(shape[2]):
                array[cx, cy, i, i] = 1
            return tf.constant(array)
        else:
            raise
    return _initializer


def spectral_normalised_init(factor=1.0):
    """Random uniform divided by its largest singular value"""
    def _initializer(shape, dtype=tf.float32):
        data = np.random.uniform(-1.0, 1.0, shape)
        svs = np.linalg.svd(data, compute_uv=False)
        return (data / (factor * np.abs(svs[0]))).astype(np.float32)
    return _initializer
