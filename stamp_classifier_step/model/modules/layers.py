"""
Custome layers for models

@author Esteban Reyes
"""
import tensorflow as tf

from parameters import constants
from parameters import errors


# ToDo: fix None in activation, because it's not callable


def dense(
    inputs,
    units,
    training,
    batchnorm=constants.BN,
    drop_rate=None,
    activation=tf.nn.relu,
    kernel_initializer=None,
    bias_initializer=None,
    name=None,
):
    """Builds a dense/fully connected layer with batch normalization and dropout.

    Parameters
    ----------
    inputs : tf.Tensor
      (2d tensor) Input tensor of shape [batch_size, n_features].
    units : int
      Number of units (neurons) to apply.
    training : tf.Tensor
      Placeholder of a boolean tensor which indicates if it is the training phase
      or not.
    batchnorm : str or None
      [{None, BN, BN_RENORM}, defaults to BN] Type of batchnorm to be used.
      BN is normal batchnorm, and BN_RENORM is a batchnorm with renorm activated.
      If None, batchnorm is not applied. The batchnorm layer is applied before
      activation.
    drop_rate : int or None
      Rate (percentage) of neuron to drop (turn-off or multiply by zero) at every
      train step. In validation phase it is not used, and amplifies output by
      1/drop_rate. If it is None is not used.
    activation : callable or None
       Type of activation to be used after convolution. If None, activation is
       linear.
    kernel_initializer : callable or None
      An initializer for the dense kernel.
    bias_initializer : callable or None
      An initializer for the dense biases.
    name : str
      A name for the operation.

    Returns
    -------
    output : tf.Tensor
      Dense layer output, that can comprehend batchnorm, activation and dropout,
      in that sequential order.
    """
    with tf.compat.v1.variable_scope(name):
        outputs = tf.keras.layers.Dense(
            units=units,
            activation=None,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            name=name,
        )(inputs)
        # Why scale is false when using ReLU as the next activation
        # https://datascience.stackexchange.com/questions/22073/
        # why-is-scale-parameter-on-batch-normalization-not-needed-on-relu/22127

        if batchnorm:  # batchnorm parameter is not None
            # Here we add a pre-activation batch norm
            if activation == tf.nn.relu:
                outputs = batchnorm_layer(
                    outputs,
                    name="bn",
                    scale=False,
                    batchnorm=batchnorm,
                    training=training,
                )
            else:
                outputs = batchnorm_layer(
                    outputs,
                    name="bn",
                    scale=True,
                    batchnorm=batchnorm,
                    training=training,
                )

        outputs = activation(outputs)

        if drop_rate is not None:
            rate = drop_rate * tf.cast(
                training, tf.float32
            )  # if training is False rate=0
            # tf.summary.scalar('drop_rate', rate)
            outputs = tf.keras.layers.Dropout(rate=rate, name="dp")(outputs)
            # tf.summary.histogram('output', output)
    return outputs


# ToDo: avoid code replication from conv2d
# ToDo: research if inputs can be just tf.Tensors or also np.arrays
# ToDo: research if for functions is better to us callable or something like
#  typing.Callable[[], None]
def deconv2d(
    inputs,
    filters,
    training,
    batchnorm=constants.BN,
    kernel_size=5,
    strides=2,
    padding=constants.PAD_SAME,
    data_format="channels_last",
    activation=tf.nn.relu,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    name=None,
):
    """Builds a 2d up sampling convolutional layer with batch normalization.

    Parameters
    ----------
    inputs : tf.Tensor
      (4d tensor) Input tensor of shape [batch_size, height, width, n_channels].
    filters : int
      Number of filters to apply.
    training : tf.Tensor
      Placeholder of a boolean tensor which indicates if it is the training phase
      or not.
    batchnorm : str or None
      [{None, BN, BN_RENORM}, defaults to BN] Type of batchnorm to be used.
      BN is normal batchnorm, and BN_RENORM is a batchnorm with renorm activated.
      If None, batchnorm is not applied. The batchnorm layer is applied before
      activation.
    kernel_size : int or tuple
      Size of the kernels.
    strides : int or tuple
      Size of the strides of the convolutions.
    padding : str
      [{PAD_SAME, PAD_VALID}, defaults to PAD_SAME] Type of padding for the
      convolution.
    activation : callable or None
       Type of activation to be used after convolution. If None, activation is
       linear.
    kernel_initializer : callable or None
      An initializer for the deconvolution kernel.
    bias_initializer : callable or None
      An initializer for the deconvolution biases.
    name : str
      A name for the operation.

    Returns
    -------
    output : tf.Tensor
      Up sampled layer output, that can comprehend batchnorm and activation.
    """
    errors.check_valid_value(
        padding, "padding", [constants.PAD_SAME, constants.PAD_VALID]
    )

    with tf.variable_scope(name):
        outputs = tf.layers.conv2d_transpose(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=None,
            padding=padding,
            kernel_initializer=kernel_initializer,
            data_format=data_format,
            bias_initializer=bias_initializer,
            name="deconv2d",
        )
        # Why scale is false when using ReLU as the next activation
        # https://datascience.stackexchange.com/questions/22073/why-is-scale-
        # parameter-on-batch-normalization-not-needed-on-relu/22127

        if batchnorm:  # batchnorm parameter is not None
            # Here we add a pre-activation batch norm
            if activation == tf.nn.relu:
                outputs = batchnorm_layer(
                    outputs,
                    name="bn",
                    scale=False,
                    batchnorm=batchnorm,
                    training=training,
                )
            else:
                outputs = batchnorm_layer(
                    outputs,
                    name="bn",
                    scale=True,
                    batchnorm=batchnorm,
                    training=training,
                )
        output = activation(outputs)
    return output


def conv2d(
    inputs,
    filters,
    training,
    batchnorm=constants.BN,
    activation=tf.nn.relu,
    padding=constants.PAD_SAME,
    data_format="channels_last",
    kernel_size=3,
    strides=1,
    kernel_initializer=None,
    name=None,
):
    """Buils a 2d convolutional layer with batch normalization.

    Args:
        inputs: (4d tensor) input tensor of shape
            [batch_size, height, width, n_channels]
        filters: (int) Number of filters to apply.
        training: (Optional, boolean, defaults to False) Indicates if it is the
            training phase or not.
        batchnorm: (Optional, {None, BN, BN_RENORM}, defaults to BN) Type of
            batchnorm to be used. BN is normal batchnorm, and BN_RENORM is a
            batchnorm with renorm activated. If None, batchnorm is not applied.
            The batchnorm layer is applied before activation.
        activation: (Optional, function, defaults to tf.nn.relu) Type of
            activation to be used after convolution. If None, activation is
            linear.
        padding: (Optional, {PAD_SAME, PAD_VALID}, defaults to PAD_SAME) Type
            of padding for the convolution.
        kernel_size: (Optional, int or tuple of int, defaults to 3) Size of
            the kernels.
        strides: (Optional, int or tuple of int, defaults to 1) Size of the
            strides of the convolutions.
        kernel_initializer: (Optional, function, defaults to None) An
            initializer for the convolution kernel.
        name: (Optional, string, defaults to None) A name for the operation.
    """
    errors.check_valid_value(
        padding, "padding", [constants.PAD_SAME, constants.PAD_VALID]
    )

    with tf.compat.v1.variable_scope(name):
        outputs = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=None,
            padding=padding,
            data_format=data_format,
            kernel_initializer=kernel_initializer,
            name="conv2d",
        )(inputs)
        # Why scale is false when using ReLU as the next activation
        # https://datascience.stackexchange.com/questions/22073/why-is-scale-
        # parameter-on-batch-normalization-not-needed-on-relu/22127

        if batchnorm:  # batchnorm parameter is not None
            # Here we add a pre-activation batch norm
            if activation == tf.nn.relu:
                outputs = batchnorm_layer(
                    outputs,
                    name="bn",
                    scale=False,
                    batchnorm=batchnorm,
                    training=training,
                )
            else:
                outputs = batchnorm_layer(
                    outputs,
                    name="bn",
                    scale=True,
                    batchnorm=batchnorm,
                    training=training,
                )
        output = activation(outputs)
    return output


def batchnorm_layer(
    inputs, name, scale=True, batchnorm=constants.BN, reuse=False, training=False
):
    """Buils a batch normalization layer.
    By default, it uses a faster, fused implementation if possible.

    Args:
        inputs: (tensor) Input tensor of shape [batch_size, ..., channels].
        name: (string) A name for the operation.
        scale: (Optional, bool, defaults to True) Whether to add the scale
            parameter.
        batchnorm: (Optional, {BN, BN_RENORM}, defaults to BN) Type of batchnorm
            to be used. BN is normal batchnorm, and BN_RENORM is a batchnorm
            with renorm activated.
        reuse: (Optional, boolean, defaults to False) Whether to reuse the layer
            variables.
        training: (Optional, boolean, defaults to False) Indicates if it is the
            training phase or not.
    """
    errors.check_valid_value(
        batchnorm, "batchnorm", [constants.BN, constants.BN_RENORM]
    )

    if batchnorm == constants.BN:
        outputs = tf.keras.layers.BatchNormalization(scale=scale, name=name)(
            inputs, training
        )
    else:  # BN_RENORM
        name = "%s_renorm" % name
        outputs = tf.layers.batch_normalization(
            inputs=inputs,
            training=training,
            scale=scale,
            reuse=reuse,
            renorm=True,
            name=name,
        )
    return outputs


def pooling_layer(inputs, pooling=constants.MAXPOOL, pool_size=2, strides=2, name=None):
    """
    Args:

        inputs: (4d tensor) input tensor of shape
                [batch_size, height, width, n_channels]
        pooling: (Optional, {AVGPOOL, MAXPOOL}, defaults to MAXPOOL) Type of
            pooling to be used, which is always of stride 2
            and pool size 2.
        pool_size: (Optional, defaults to 2) pool size of pooling operation
        strides: (Optional, defaults to 2) strides of pooling operation
        name: (Optional, defaults to None) A name for the operation.
    """
    errors.check_valid_value(pooling, "pooling", [constants.AVGPOOL, constants.MAXPOOL])

    if pooling == constants.AVGPOOL:
        outputs = tf.layers.average_pooling2d(
            inputs=inputs, pool_size=pool_size, strides=strides, name=name
        )
    else:  # MAXPOOL
        outputs = tf.keras.layers.MaxPooling2D(
            pool_size=pool_size, strides=strides, name=name
        )(inputs)
    return outputs


def upsampling_layer(inputs, filters, padding=constants.PAD_SAME, name=None):
    errors.check_valid_value(
        padding, "padding", [constants.PAD_SAME, constants.PAD_VALID]
    )

    outputs = tf.layers.conv2d_transpose(
        inputs=inputs,
        filters=filters,
        kernel_size=2,
        strides=2,
        activation=None,
        padding=padding,
        name=name,
    )
    return outputs


def cyclic_avg_pool(inputs, name=""):
    with tf.name_scope(name):
        _, in_feat = inputs.get_shape().as_list()
        separate_rotations = tf.reshape(inputs, [4, -1, in_feat])
        return tf.reduce_mean(separate_rotations, axis=0)
