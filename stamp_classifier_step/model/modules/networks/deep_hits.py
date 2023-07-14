from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from modules import layers
from parameters import param_keys
from modules.networks.simple_cnn import SimpleCNN
from parameters import constants


class DeepHits(SimpleCNN):
    def __init__(self, inputs, params, training_flag):
        super().__init__(inputs, params, training_flag)

    def _build_network(self, inputs, params, training_flag):
        batchnorm_conv = params[param_keys.BATCHNORM_CONV]
        batchnorm_fc = params[param_keys.BATCHNORM_FC]
        drop_rate = params[param_keys.DROP_RATE]
        kernel_shape = params[param_keys.KERNEL_SIZE]
        pool_shape = params[param_keys.POOL_SIZE]

        padded_inputs = tf.pad(
            inputs, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]], name="padding"
        )

        conv_1 = layers.conv2d(
            inputs=padded_inputs,
            filters=32,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=4,
            padding=constants.PAD_VALID,
            name="conv_1_1",
        )
        conv_2 = layers.conv2d(
            inputs=conv_1,
            filters=32,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=kernel_shape,
            name="conv_2_2",
        )
        down_1 = layers.pooling_layer(
            conv_2, pool_size=pool_shape, strides=pool_shape, name="down_1_3"
        )

        conv_3 = layers.conv2d(
            inputs=down_1,
            filters=64,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=kernel_shape,
            name="conv_3_4",
        )
        conv_4 = layers.conv2d(
            inputs=conv_3,
            filters=64,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=kernel_shape,
            name="conv_4_5",
        )
        conv_5 = layers.conv2d(
            inputs=conv_4,
            filters=64,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=kernel_shape,
            name="conv_5_6",
        )
        down_2 = layers.pooling_layer(
            conv_5, pool_size=pool_shape, strides=pool_shape, name="down_2_7"
        )

        flatten = tf.layers.flatten(inputs=down_2, name="flatten_1_8")

        dense_1 = layers.dense(
            inputs=flatten,
            units=64,
            training=training_flag,
            batchnorm=batchnorm_fc,
            drop_rate=drop_rate,
            name="dense_1_9",
        )

        dense_2 = layers.dense(
            inputs=dense_1,
            units=64,
            training=training_flag,
            batchnorm=batchnorm_fc,
            drop_rate=drop_rate,
            name="dense_2_10",
        )

        cyclic_avg_pool = layers.cyclic_avg_pool(dense_2, name="cyclic_avg_pool_1_11")

        output = tf.layers.dense(
            inputs=cyclic_avg_pool,
            units=params[param_keys.NUMBER_OF_CLASSES],
            activation=None,
            name="output_logits",
        )

        self._layers_list = [
            conv_1,
            conv_2,
            down_1,
            conv_3,
            conv_4,
            conv_5,
            down_2,
            flatten,
            dense_1,
            dense_2,
            cyclic_avg_pool,
            output,
        ]

        return output

    def get_last_conv_layer(self):
        return self._layers_list[6]
