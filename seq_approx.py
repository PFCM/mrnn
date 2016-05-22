"""Test out the guys at approximating sequences"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from mrnn import tensor_ops as tops


def get_tt_model(inputs, shape, initial_states,
                 trainable=True, name='TT'):
    """Gets a recurrent tensor train model.
    
    Args:
      inputs: list of tensors -- the input sequence in time major order.
        (ie. a list of tensors each of shape `[batch_size x input_size]`.
      shape: the shape of the model (how many layers, the shape of each
        layer. Should be a sequence of sequences, one for each layer and
        each one consisting of (input_size, output_size, hidden_size, 
        rank1, rank2).
      initial_states: initial hidden states. Should be one per layer of
        appropriate size.
      trainable: whether the variables are added to the tensorflow
        collection of trainable variables (and hence whether they will
        be optimised or not).
    
    Returns:
      list of tensors: the outputs of the net.
    """
    with tf.variable_scope(name) as scope:
        outputs = []
        states = [initial_states]
        for step, net_in in enumerate(inputs):
            output = net_in
            # do the layers
            for i, layer in enumerate(shape):
                tensor = tops.get_tt_3_tensor(layer[:3], layer[3:],
                                              trainable=trainable)
                output = tops.bilinear_product_tt_3(output, tensor,
                                                    states[step][i])
                # add output to states
                # in fact, don't really need list of outputs, just states
