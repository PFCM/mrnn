"""Train some RNNS for sequential prediction on JSB"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import shutil

from six.moves import xrange

import numpy as np
import tensorflow as tf

import mrnn
import mrnn.init as init
import rnndatasets.jsbchorales as jsb

import ptb_test  # may as well reuse


if __name__ == '__main__':
    tf.app.run()
