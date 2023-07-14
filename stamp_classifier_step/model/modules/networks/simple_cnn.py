from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from modules import layers
from parameters import param_keys


class SimpleCNN(object):
    def __init__(self, inputs, params, training_flag):
        self.logits = self._build_network(inputs, params, training_flag)

    def _build_network(self, inputs, params, training_flag):
        batchnorm_conv = params[param_keys.BATCHNORM_CONV]
        batchnorm_fc = params[param_keys.BATCHNORM_FC]
        drop_rate = params[param_keys.DROP_RATE]
        kernel_shape = params[param_keys.KERNEL_SIZE]
        pool_shape = params[param_keys.POOL_SIZE]

        conv_1 = layers.conv2d(
            inputs=inputs,
            filters=32,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=kernel_shape,
            name="conv_1_1",
        )
        down_1 = layers.pooling_layer(
            conv_1, pool_size=pool_shape, strides=pool_shape, name="down_1_2"
        )

        conv_2 = layers.conv2d(
            inputs=down_1,
            filters=32,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=kernel_shape,
            name="conv_2_1",
        )
        down_2 = layers.pooling_layer(
            conv_2, pool_size=pool_shape, strides=pool_shape, name="down_2_2"
        )

        conv_3 = layers.conv2d(
            inputs=down_2,
            filters=64,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=kernel_shape,
            name="conv_3_1",
        )
        down_3 = layers.pooling_layer(
            conv_3, pool_size=pool_shape, strides=pool_shape, name="down_3_2"
        )

        conv_4 = layers.conv2d(
            inputs=down_3,
            filters=64,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=kernel_shape,
            name="conv_4_1",
        )
        down_4 = layers.pooling_layer(
            conv_4, pool_size=pool_shape, strides=pool_shape, name="down_4_2"
        )

        conv_5 = layers.conv2d(
            inputs=down_4,
            filters=128,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=kernel_shape,
            name="conv_5_1",
        )
        down_5 = layers.pooling_layer(
            conv_5, pool_size=pool_shape, strides=pool_shape, name="down_5_2"
        )

        flatten = tf.layers.flatten(inputs=down_5, name="flatten")

        dense_1 = layers.dense(
            inputs=flatten,
            units=256,
            training=training_flag,
            batchnorm=batchnorm_fc,
            drop_rate=drop_rate,
            name="dense_6_1",
        )

        dense_2 = layers.dense(
            inputs=dense_1,
            units=128,
            training=training_flag,
            batchnorm=batchnorm_fc,
            drop_rate=drop_rate,
            name="dense_7_1",
        )

        output = tf.layers.dense(
            inputs=dense_2,
            units=params[param_keys.NUMBER_OF_CLASSES],
            activation=None,
            name="output_logits",
        )

        self._layers_list = [
            conv_1,
            down_1,
            conv_2,
            down_2,
            conv_3,
            conv_4,
            down_4,
            conv_5,
            down_5,
            flatten,
            dense_1,
            dense_2,
            output,
        ]

        return output

    def get_last_conv_layer(self):
        return self._layers_list[8]

    def get_layer_before_logits(self):
        return self._layers_list[-2]

    def get_output(self):
        return self.logits
