from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras.initializers import _compute_fans
from keras.layers import *
from keras.models import Model
from keras.regularizers import l2


from modules import layers
from modules.networks.simple_cnn import SimpleCNN
from parameters import param_keys, constants

WEIGHT_DECAY = 0.5 * 0.0005


def _get_channels_axis():
    return -1 if K.image_data_format() == "channels_last" else 1


_conv_kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
    factor=2.0, mode=constants.FAN_IN, uniform=False
)


_dense_kernel_initializer = tf.contrib.layers.variance_scaling_initializer(
    factor=1.0, mode=constants.FAN_IN, uniform=True
)


def batch_norm(inputs, training):
    return tf.layers.batch_normalization(
        inputs=inputs,
        training=training,
        momentum=0.9,
        epsilon=1e-5,
        beta_regularizer=l2(WEIGHT_DECAY),
        gamma_regularizer=l2(WEIGHT_DECAY),
    )


def conv2d(inputs, output_channels, kernel_size, strides=1):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=output_channels,
        kernel_size=kernel_size,
        strides=strides,
        kernel_initializer=_conv_kernel_initializer,
        kernel_regularizer=l2(WEIGHT_DECAY),
        padding="same",
    )


def dense(inputs, output_units):
    return tf.layers.dense(
        inputs=inputs,
        units=output_units,
        kernel_initializer=_dense_kernel_initializer,
        kernel_regularizer=l2(WEIGHT_DECAY),
        bias_regularizer=l2(WEIGHT_DECAY),
    )


def drop_out(inputs, drop_rate, training):
    if drop_rate is not None:
        rate = drop_rate * tf.cast(training, tf.float32)  # if training is False rate=0
        # tf.summary.scalar('drop_rate', rate)
        outputs = tf.layers.dropout(inputs, rate=rate, name="dp")
        # tf.summary.histogram('output', output)
    else:
        outputs = inputs
    return outputs


def _add_basic_block(x_in, training, out_channels, strides, dropout_rate=0.0):
    is_channels_equal = K.int_shape(x_in)[_get_channels_axis()] == out_channels
    bn1 = batch_norm(x_in, training)

    bn1 = tf.nn.relu(bn1)
    out = conv2d(bn1, out_channels, 3, strides)
    out = batch_norm(out, training)
    out = tf.nn.relu(out)
    out = drop_out(out, dropout_rate, training)
    out = conv2d(out, out_channels, 3, 1)
    shortcut = x_in if is_channels_equal else conv2d(bn1, out_channels, 1, strides)
    out = out + shortcut
    return out


def _add_conv_group(x_in, training, out_channels, n, strides, dropout_rate=0.0):
    out = _add_basic_block(x_in, training, out_channels, strides, dropout_rate)
    for _ in range(1, n):
        out = _add_basic_block(out, training, out_channels, 1, dropout_rate)
    return out


def create_wide_residual_network(
    inputs, training, depth, widen_factor=1, dropout_rate=0.0
):
    n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
    assert (depth - 4) % 6 == 0, "depth should be 6n+4"
    n = (depth - 4) // 6

    with tf.variable_scope("conv_1"):
        conv1 = conv2d(
            inputs, n_channels[0], 3
        )  # one conv at the beginning (spatial size: 32x32)
    with tf.variable_scope("block_1"):
        conv2 = _add_conv_group(
            conv1, training, n_channels[1], n, 1, dropout_rate
        )  # Stage 1 (spatial size: 32x32)
    with tf.variable_scope("block_2"):
        conv3 = _add_conv_group(
            conv2, training, n_channels[2], n, 2, dropout_rate
        )  # Stage 2 (spatial size: 16x16)
    with tf.variable_scope("block_3"):
        conv4 = _add_conv_group(
            conv3, training, n_channels[3], n, 2, dropout_rate
        )  # Stage 3 (spatial size: 8x8)
    with tf.variable_scope("wide_out"):
        out = batch_norm(conv4, training)
        out = tf.nn.relu(out)
        out = tf.keras.layers.GlobalAveragePooling2D()(out)  # AveragePooling2D()(out)
    # out = Flatten()(out)

    # out = dense(num_classes)(out)
    # out = Activation(final_activation)(out)

    return out


class WideNet(SimpleCNN):
    def __init__(self, inputs, params, training_flag):
        super().__init__(inputs, params, training_flag)

    def _build_network(self, inputs, params, training_flag):
        batchnorm_fc = params[param_keys.BATCHNORM_FC]
        drop_rate = params[param_keys.DROP_RATE]

        n, k = (10, 4)
        with tf.variable_scope("wide_net"):
            mdl_output = create_wide_residual_network(inputs, training_flag, n, k)
        with tf.variable_scope("hits_dense"):
            dense_1 = layers.dense(
                inputs=mdl_output,
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

            cyclic_avg_pool = layers.cyclic_avg_pool(
                dense_2, name="cyclic_avg_pool_1_11"
            )

            output = tf.layers.dense(
                inputs=cyclic_avg_pool,
                units=params[param_keys.NUMBER_OF_CLASSES],
                activation=None,
                name="output_logits",
            )

        return output
