"""quick tests just to make sure that it lines up"""
import unittest

import numpy as np
import tensorflow as tf

import mrnn.tensor_ops as tops


class TensorOpsTests(unittest.TestCase):
    """Fairly simple tests, making sure the dims line up etc"""

    def setUp(self):
        """starts up a tf session"""
        self.sess = tf.InteractiveSession()

    def test_get_cp(self):
        """Make sure we can get a tensor in cp form as expected"""
        shape = [11, 12, 13]
        rank = 10
        tensor = tops.get_cp_tensor(shape, rank, 'test')
        self.assertEqual(len(tensor), len(shape))
        for dim_1, mat in zip(shape, tensor):
            self.assertEqual(mat.get_shape(), [rank, dim_1])

    def test_bilinear_cp_shape(self):
        """Make sure the thing actually gives us the shapes we want"""
        # get some shapes
        with tf.variable_scope('tests',
                               initializer=tf.random_normal_initializer()):
            cp_mat = tops.get_cp_tensor([12, 13, 14], 5, 'test')
            x = tf.get_variable('x', [1, 12], dtype=tf.float32)
            y = tf.get_variable('y', [1, 14], dtype=tf.float32)
            result = tops.bilinear_product_cp(x, cp_mat, y)

        self.sess.run(tf.initialize_all_variables())
        answer = self.sess.run(result)
        self.assertEqual(answer.shape, (1, 13))
