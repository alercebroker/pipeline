from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from parameters import general_keys


# TODO: transform metrics to a class in order to easily return metric dict


def accuracy(labels, predictions, is_onehot=False):
    with tf.name_scope("accuracy"):
        if is_onehot:
            labels = tf.argmax(labels, axis=-1)
            predictions = tf.argmax(predictions, axis=-1)
        correct_predictions = tf.equal(labels, predictions)
        casted_correct_pred = tf.cast(correct_predictions, tf.float32)
    metrics_dict = {general_keys.ACCURACY: casted_correct_pred}
    return metrics_dict


def flatten(inputs, name=None):
    """Flattens [batch_size, d0, ..., dn] to [batch_size, d0*...*dn]"""
    with tf.name_scope(name):
        dim = tf.reduce_prod(tf.shape(inputs)[1:])
        outputs = tf.reshape(inputs, shape=(-1, dim))
    return outputs


def iou_and_dice_deprecated(labels, predictions, onehot_format=False):
    """Computes IoU and DICE metrics of myocardial segmentation.
    If onehot_format is False, then labels and predictions should have shape
    [batch, height, width], and be integer values in the range (0, 1, 2).
    If onehot_hot is True, the data is assumed to be already in onehot and a
    shape of [batch, height, width, n_classes] of binary numbers is required.
    Returns the metrics for each image.
    """
    if onehot_format:
        predictions_onehot = predictions
        labels_onehot = labels
    else:
        predictions_onehot = tf.one_hot(predictions, 3)
        labels_onehot = tf.one_hot(labels, 3)

    predictions_class_1 = predictions_onehot[..., 1]
    labels_class_1 = labels_onehot[..., 1]

    predictions_flatten = flatten(predictions_class_1)
    labels_flatten = flatten(labels_class_1)

    inter_area = tf.reduce_sum(tf.multiply(predictions_flatten, labels_flatten), axis=1)
    sum_area = tf.reduce_sum(predictions_flatten, axis=1) + tf.reduce_sum(
        labels_flatten, axis=1
    )
    union_area = sum_area - inter_area
    iou = inter_area / union_area
    dice = 2 * inter_area / sum_area
    return iou, dice


def iou_and_dice_v2(labels, predictions, is_onehot=True, average=True):
    """
    Computes IoU and DICE metrics of myocardial segmentation.
    If is_onehot is False, then labels and predictions should have shape
    [batch, height, width], and be integer values in the range (0, 1, 2).
    If is_onehot is True, the data is assumed to be already in onehot and a
    shape of [batch, height, width, n_classes] of binary numbers is required.
    If average is true, then the mean value of the metrics along the batch is
    returned. If false, each a tensor of shape (batch,) that contains the value
    of the metric in each example is returned.
    Returns the metrics in a dictionary.
    """
    if is_onehot:
        labels_onehot = labels
        predictions_onehot = predictions
    else:
        labels_onehot = tf.one_hot(labels, 3)
        predictions_onehot = tf.one_hot(predictions, 3)

    # Endocardial metrics: Here we consider the entire area inside the
    # endocardial border, so only class 2 is used.
    labels_endo = labels_onehot[..., 2]
    predictions_endo = predictions_onehot[..., 2]
    inter_area_endo, sum_area_endo = inter_and_sum_areas(labels_endo, predictions_endo)

    # Epicardial metrics: Here we consider the entire area inside the
    # epicardial border, so classes 1 and 2 are used together.
    labels_epi = labels_onehot[..., 1] + labels_onehot[..., 2]
    predictions_epi = predictions_onehot[..., 1] + predictions_onehot[..., 2]
    inter_area_epi, sum_area_epi = inter_and_sum_areas(labels_epi, predictions_epi)

    # Myocardial metrics: Here we consider the area between the epicardial
    # and endocardial borders, so only class 1 is used.
    labels_myo = labels_onehot[..., 1]
    predictions_myo = predictions_onehot[..., 1]
    inter_area_myo, sum_area_myo = inter_and_sum_areas(labels_myo, predictions_myo)

    # Compute metrics
    metrics_dict = {
        general_keys.DICE_ENDO: 2 * inter_area_endo / sum_area_endo,
        general_keys.IOU_ENDO: inter_area_endo / (sum_area_endo - inter_area_endo),
        general_keys.DICE_EPI: 2 * inter_area_epi / sum_area_epi,
        general_keys.IOU_EPI: inter_area_epi / (sum_area_epi - inter_area_epi),
        general_keys.DICE_MYO: 2 * inter_area_myo / sum_area_myo,
        general_keys.IOU_MYO: inter_area_myo / (sum_area_myo - inter_area_myo),
    }

    if average:
        for key in metrics_dict:
            metrics_dict[key] = tf.reduce_mean(metrics_dict[key])
    return metrics_dict


def inter_and_sum_areas(labels, predictions):
    """
    labels and predictions are assumed to be a batch of binary images.
    shape [batch, height, width].
    """
    predictions_flatten = flatten(predictions)
    labels_flatten = flatten(labels)
    inter_area = tf.reduce_sum(tf.multiply(predictions_flatten, labels_flatten), axis=1)
    sum_area = tf.reduce_sum(predictions_flatten, axis=1) + tf.reduce_sum(
        labels_flatten, axis=1
    )
    return inter_area, sum_area
