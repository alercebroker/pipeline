from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from modules import layers
from parameters import param_keys
from modules.networks.simple_cnn import SimpleCNN
from parameters import constants


class DeepHitsConvDecoder(SimpleCNN):
    def __init__(self, inputs, params, training_flag):
        super().__init__(inputs, params, training_flag)

    def _build_network(self, inputs, params, training_flag):
        batchnorm_conv = params[param_keys.BATCHNORM_CONV]
        kernel_shape = params[param_keys.KERNEL_SIZE]
        pool_shape = params[param_keys.POOL_SIZE]
        img_size = params[param_keys.INPUT_IMAGE_SIZE]

        deconv_1 = layers.deconv2d(
            inputs,
            filters=64,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=kernel_shape,
            strides=pool_shape,
            activation=tf.nn.relu,
            name="deconv_1_8",
        )
        conv_1 = layers.conv2d(
            inputs=deconv_1,
            filters=64,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=kernel_shape,
            name="conv_6_9",
        )
        conv_2 = layers.conv2d(
            inputs=conv_1,
            filters=64,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=kernel_shape,
            name="conv_7_10",
        )

        deconv_2 = layers.deconv2d(
            conv_2,
            filters=32,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=kernel_shape,
            strides=pool_shape,
            activation=tf.nn.relu,
            name="deconv_2_11",
        )
        conv_3 = layers.conv2d(
            inputs=deconv_2,
            filters=32,
            training=training_flag,
            batchnorm=batchnorm_conv,
            kernel_size=4,
            name="conv_8_12",
        )

        reshape_output = conv_3[:, :img_size, :img_size, :]

        conv_4 = layers.conv2d(
            inputs=reshape_output,
            filters=1,
            training=training_flag,
            batchnorm=None,
            kernel_size=4,
            activation=tf.nn.sigmoid,
            name="conv_9_13",
        )

        with tf.name_scope("un_rotate_1_14"):
            rot0, rot90, rot180, rot270 = tf.split(conv_4, num_or_size_splits=4, axis=0)
            desrot90 = tf.image.rot90(rot90, k=3)
            desrot180 = tf.image.rot90(rot180, k=2)
            desrot270 = tf.image.rot90(rot270, k=1)
            all_rot = tf.stack([rot0, desrot90, desrot180, desrot270], axis=0)
            merge_rotations = tf.reduce_mean(all_rot, axis=0)

        output = merge_rotations

        return output
