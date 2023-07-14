from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from modules import layers
from parameters import param_keys
from modules.networks.simple_cnn import SimpleCNN
from parameters import constants


class LargeDeepHits(SimpleCNN):
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
            kernel_size=4,
            padding=constants.PAD_VALID,
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
        down_2 = layers.pooling_layer(
            conv_4, pool_size=pool_shape, strides=pool_shape, name="down_2_6"
        )

        conv_5 = layers.conv2d(
            inputs=down_2,
            filters=128,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=kernel_shape,
            name="conv_5_7",
        )
        conv_6 = layers.conv2d(
            inputs=conv_5,
            filters=128,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=kernel_shape,
            name="conv_6_8",
        )
        conv_7 = layers.conv2d(
            inputs=conv_6,
            filters=128,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=kernel_shape,
            name="conv_7_9",
        )
        down_3 = layers.pooling_layer(
            conv_7, pool_size=pool_shape, strides=pool_shape, name="down_3_10"
        )

        flatten = tf.layers.flatten(inputs=down_3, name="flatten_1_11")

        dense_1 = layers.dense(
            inputs=flatten,
            units=128,
            training=training_flag,
            batchnorm=batchnorm_fc,
            drop_rate=drop_rate,
            name="dense_1_12",
        )

        dense_2 = layers.dense(
            inputs=dense_1,
            units=128,
            training=training_flag,
            batchnorm=batchnorm_fc,
            drop_rate=drop_rate,
            name="dense_2_13",
        )

        cyclic_avg_pool = layers.cyclic_avg_pool(dense_2, name="cyclic_avg_pool_1_14")

        output = tf.layers.dense(
            inputs=cyclic_avg_pool,
            units=params[param_keys.NUMBER_OF_CLASSES],
            activation=None,
            name="output_logits",
        )

        return output
