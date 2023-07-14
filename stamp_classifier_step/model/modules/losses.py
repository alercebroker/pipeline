from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from parameters import param_keys


def wasserstein_gradient_penalty(
    params, disc_fake, disc_real, interpolates, disc_interpolates
):
    with tf.name_scope("gen_loss"):
        gen_loss = -tf.reduce_mean(disc_fake)
        tf.summary.scalar("losses/gen_loss", gen_loss)
    with tf.name_scope("disc_loss"):
        disc_loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        tf.summary.scalar("losses/disc_loss", disc_loss)
        with tf.name_scope("gradient_penalty"):
            gradients = tf.gradients(disc_interpolates, [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
            tf.summary.scalar("penalties/gradient_penalty", gradient_penalty)
        disc_loss_gp = disc_loss + params[param_keys.LAMBDA] * gradient_penalty
        tf.summary.scalar("losses/disc_loss_gp", disc_loss_gp)
    return gen_loss, disc_loss_gp


def dcgan_min_max(disc_fake, disc_real):
    with tf.name_scope("gen_loss"):
        gen_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=disc_fake, labels=tf.ones_like(disc_fake)
            )
        )
        tf.summary.scalar("losses/gen_loss", gen_loss)
    with tf.name_scope("disc_loss"):
        disc_loss_first_term = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=disc_fake, labels=tf.zeros_like(disc_fake)
            )
        )
        tf.summary.scalar("losses/disc_loss_1", disc_loss_first_term)
        disc_loss_second_term = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=disc_real, labels=tf.ones_like(disc_real)
            )
        )
        tf.summary.scalar("losses/disc_loss_2", disc_loss_second_term)
        disc_loss = (disc_loss_first_term + disc_loss_second_term) / 2.0
        tf.summary.scalar("losses/disc_loss", disc_loss)
    return gen_loss, disc_loss


def xentropy(logits, input_labels, number_of_classes):
    input_labels_oh = tf.one_hot(input_labels, number_of_classes)
    # TODO: ask why these reshapes
    flat_logits = tf.reshape(tensor=logits, shape=(-1, number_of_classes))
    flat_labels = tf.cast(
        tf.reshape(tensor=input_labels_oh, shape=(-1, number_of_classes)), tf.float32
    )
    diff = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=flat_logits, labels=flat_labels
    )
    loss = tf.reduce_mean(diff)
    return loss


def entropy_regularization(logits):
    class_distr = tf.nn.softmax(logits, name="entropy_regularization_softmax")
    entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=class_distr, logits=logits, name="entropy_loss"
    )
    mean_entropy = tf.reduce_mean(entropy_loss, name="mean_entropy")
    tf.compat.v1.summary.scalar("entropy_regularizer", mean_entropy)
    return mean_entropy


def xentropy_with_entropy_regularization(
    logits, input_labels, number_of_classes, beta=1e-3
):
    cross_entropy = xentropy(logits, input_labels, number_of_classes)
    with tf.name_scope("entropy_regularization"):
        entropy_loss = entropy_regularization(logits)
    return tf.identity(
        cross_entropy - beta * entropy_loss, name="xentropy_with_entropy_reg"
    )


def reconstruction_error(original, reconstruction):
    return tf.reduce_mean(tf.square(tf.subtract(original, reconstruction)))
