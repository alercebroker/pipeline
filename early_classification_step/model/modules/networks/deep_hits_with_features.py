from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from modules import layers
from parameters import param_keys
from modules.networks.simple_cnn import SimpleCNN
from parameters import constants


class DeepHitsWithFeatures(SimpleCNN):
    def __init__(self, inputs, features, params, training_flag):
        self.logits = self._build_network(inputs, features, params, training_flag)

    def _build_network(self, inputs, features, params, training_flag):
        batchnorm_conv = params[param_keys.BATCHNORM_CONV]
        batchnorm_fc = params[param_keys.BATCHNORM_FC]
        drop_rate = params[param_keys.DROP_RATE]
        kernel_shape = params[param_keys.KERNEL_SIZE]
        pool_shape = params[param_keys.POOL_SIZE]
        batchnorm_features_fc = params[param_keys.BATCHNORM_FEATURES_FC]

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

        flatten = tf.keras.layers.Flatten(name="flatten_1_8")(down_2)

        dense_1 = layers.dense(
            inputs=flatten,
            units=64,
            training=training_flag,
            batchnorm=batchnorm_fc,
            drop_rate=drop_rate,
            name="dense_1_10",
        )

        cyclic_avg_pool = layers.cyclic_avg_pool(dense_1, name="cyclic_avg_pool_1_11")

        if batchnorm_features_fc:
            normalized_features = layers.batchnorm_layer(
                features, name="batchnorm_features", training=training_flag
            )
        else:
            normalized_features = features

        features_for_each_rotation = tf.concat(
            [cyclic_avg_pool, normalized_features], axis=1, name="dense_with_features"
        )

        dense_2 = layers.dense(
            inputs=features_for_each_rotation,
            units=64,
            training=training_flag,
            batchnorm=batchnorm_fc,
            drop_rate=drop_rate,
            name="dense_2_12",
        )

        dense_3 = layers.dense(
            inputs=dense_2,
            units=64,
            training=training_flag,
            batchnorm=batchnorm_fc,
            drop_rate=drop_rate,
            name="dense_3_13",
        )

        output = tf.keras.layers.Dense(
            units=params[param_keys.NUMBER_OF_CLASSES],
            activation=None,
            name="output_logits",
        )(dense_3)

        return output
