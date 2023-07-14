from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# ToDo: carefull with this, wont be possible if there are many model instances
def get_trainable_params_by_scope(scope_name):
    if scope_name is None:
        return None
    trainable_params = []
    for var in tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="%s" % scope_name
    ):
        trainable_params.append(var)
    # print(trainable_params)
    return trainable_params


# ToDo: check if need specific scope name to update ops (Example in comments),
#  to not update BN of other network, like Gen or Disc
def generic_minimizer(optimizer, loss, scope_name=None):
    var_list = get_trainable_params_by_scope(scope_name)
    update_ops = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.UPDATE_OPS, scope=scope_name
    )  # For BN ,scope='G')
    with tf.control_dependencies(update_ops):
        train_step = optimizer.minimize(loss, var_list=var_list)
    reset_optimizer_op = tf.compat.v1.variables_initializer(optimizer.variables())
    return train_step, reset_optimizer_op


def create_non_trainable(initial_value, name):
    return tf.Variable(initial_value, trainable=False, name=name)


def adam(loss, learning_rate_value, beta1=0.9, beta2=0.999, scope_name=None):
    # TODO: Learning rate from params,
    #  caution with learning rate exponential decay that is currently implemented)
    # learning rate that can be actualized through training
    learning_rate = create_non_trainable(learning_rate_value, "learning_rate")
    # Adam
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate, beta1, beta2)
    # train operation to be run for performing a learning iteration
    train_step, _ = generic_minimizer(optimizer, loss, scope_name)
    return train_step, learning_rate


def sgd(loss, learning_rate_value):
    # learning rate that can be actualized through training
    learning_rate = create_non_trainable(learning_rate_value, "learning_rate")
    # SDG optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # train operation to be run for performing a learning iteration
    train_step, _ = generic_minimizer(optimizer, loss)
    return train_step, learning_rate


def momentum_sgd(loss, learning_rate_value, momentum_value=0.5):
    # learning and momentum rate that can be actualized through training
    learning_rate = create_non_trainable(learning_rate_value, "learning_rate")
    momentum = create_non_trainable(momentum_value, "momentum")
    # MomentumSDG optimizer
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=learning_rate, momentum=momentum
    )
    # train operation to be run for performing a learning iteration
    train_step, _ = generic_minimizer(optimizer, loss)
    return train_step, learning_rate, momentum
