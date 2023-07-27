"""
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c) 2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf


def weights(
    weights_shape, initializer=tf.truncated_normal_initializer(stddev=0.05), name=""
):
    return tf.Variable(initializer(shape=weights_shape), name="weights")
    # weights_shape = weights_shape
    # return tf.Variable(tf.truncated_normal(weights_shape, stddev=0.05), name = 'weights')
    # return tf.get_variable(weights_shape,
    #                           initializer = tf.contrib.layers.xavier_initializer_conv2d())


def biases(bias_shape, initializer=tf.constant_initializer(0.05), name=""):
    return tf.Variable(initializer(shape=[bias_shape]), name="biases")
    # bias_shape = bias_shape
    # return tf.Variable(tf.constant(0.05, shape=[bias_shape]), name = 'biases')
    # return tf.get_variable([bias_shape],
    #                         initializer=initializer)
