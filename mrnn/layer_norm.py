"""A wrapper which does layer normalisation on the outputs of a cell."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import mrnn.handy_ops as hops


def basiclstm_state_splitter(output, state):
    """splitter for the states generated by the BasicLSTMCell.
    Normalises both the output and the states.
    """
    return state


def basiclstm_state_combiner(normalised):
    """reproduces the BasicLSTMCell outputs post-normalisation."""
    return normalised[1], tf.nn.rnn_cell.LSTMStateTuple(*normalised)


class LayerNormWrapper(tf.nn.rnn_cell.RNNCell):
    """Does layer normalisation to a cell. Optionally does it to the states
    instead (or more likely, both)"""

    def __init__(self, cell, separate_states=False,
                 state_splitter=basiclstm_state_splitter,
                 state_combiner=basiclstm_state_combiner):
        """Wraps a cell.

        Args:
            cell: the cell to wrap, an rnn_cell.RNNCell
            separate_states: whether the cell being wrapped has any states
                that are different from its output. If False we will ignore
                the states returned by the cell and just return the normalised
                outputs twice. If True, life gets harder.
            state_splitter: If separate states, then we need a way of getting
                the independent sets of activations that require normalisation.
                Should be callable and take the results of calling the cell,
                returning a sequence of tensors.
            state_combiner: If separate states, we need a way to recombine them
                after normalisation. Should be callable and take a list of
                tensors (in the order produced by state_splitter) and recombine
                appropriately. This is expected to return an (output, state)
                tuple we can return directly.
        """
        self._cell = cell
        self._separate_states = separate_states
        self._state_splitter = state_splitter
        self._state_combiner = state_combiner

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def input_size(self):
        return self._cell.input_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, states, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            output, states = self.cell(inputs, states, scope)
            if self._separate_states:
                to_norm = [output]
            else:
                to_norm = self._state_splitter(output, states)

            normed = [hops.layer_normalise(act) for act in to_norm]

            if self._separate_states:
                return self._state_combiner(normed)
            return tuple(normed * 2)

    def zero_state(self, batch_size, dtype):
        """Pass it on to wrapped cell in case it has been overridden"""
        return self._cell.zero_state(batch_size, dtype)
