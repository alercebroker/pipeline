#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unet with batchnormalization layers as in https://github.com/ankurhanda/tf-unet/blob/master/UNet.py
@author Esteban Reyes
"""

# python 2 and 3 compatibility
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# basic libraries
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_nans_norm_crop_stamp_model import (
    DeepHiTSNanNormCropStampModel,
)
from modules.lrp_modules.sequential import Sequential
from modules.lrp_modules.convolution import Convolution
from modules.lrp_modules.maxpool import MaxPool
from modules.lrp_modules.linear import Linear
from modules.lrp_modules.avgCyclicPool2 import CyclicAvgPool
from modules.lrp_modules.relevance_plotter import LRP_plot_tools
from parameters import param_keys


# TODO: refactor train test and validate
class DeepHiTSLRPModel(DeepHiTSNanNormCropStampModel):
    """
    Constructor
    """

    def __init__(self, params={}, model_name="DeepHitsLRP", session=None):
        super().__init__(params, model_name, session)
        (
            self.lrp_net,
            self.lrp_logits,
            self.lrp_output_probabilities,
            self.lrp_output_pred_cls,
        ) = self._build_lrp_graph(self.input_batch)
        (
            self.logit2LRP,
            self.rotated_relevance,
            _,
            self.mean_relevance,
        ) = self._init_lrp_visualization_graph()
        self.plot_tools = LRP_plot_tools()
        self._variables_init()

        # keep_prob rates of dropout set to 1, because lrp model MUST be use for
        # inference only

    def _init_lrp_model(self):
        return Sequential(
            [
                Convolution(
                    kernel_size=4,
                    output_depth=32,
                    input_depth=3,
                    act="relu",
                    stride_size=1,
                    pad="VALID",
                ),
                Convolution(
                    kernel_size=3,
                    output_depth=32,
                    stride_size=1,
                    act="relu",
                    pad="SAME",
                ),
                MaxPool(),
                Convolution(
                    kernel_size=3,
                    output_depth=64,
                    stride_size=1,
                    act="relu",
                    pad="SAME",
                ),
                Convolution(
                    kernel_size=3,
                    output_depth=64,
                    stride_size=1,
                    act="relu",
                    pad="SAME",
                ),
                Convolution(
                    kernel_size=3,
                    output_depth=64,
                    stride_size=1,
                    act="relu",
                    pad="SAME",
                ),
                MaxPool(),
                Linear(64, act="relu", keep_prob=1.0, use_dropout=True),
                Linear(64, act="relu", keep_prob=1.0, use_dropout=True),
                CyclicAvgPool(),
                Linear(self.params[param_keys.NUMBER_OF_CLASSES], act="linear"),
            ]
        )

    def _build_lrp_graph(self, input_batch):
        # TODO: see if can change for variable scope and load graph diretly by
        # changing names
        with tf.variable_scope("lrp_model"):
            lrp_net = self._init_lrp_model()
            # input batch MUST be already rotated
            padded_inputs = tf.pad(
                input_batch, paddings=[[0, 0], [3, 3], [3, 3], [0, 0]], name="padding"
            )
            lrp_logits = lrp_net.forward(padded_inputs)
        with tf.name_scope("lrp_outputs"):
            lrp_output_probabilities = tf.nn.softmax(lrp_logits)
            lrp_output_pred_cls = tf.argmax(lrp_output_probabilities, 1)
        return lrp_net, lrp_logits, lrp_output_probabilities, lrp_output_pred_cls

    def save_checkpoint_weights_to_npy(self, chkp_path):
        save_folder = os.path.join(PROJECT_PATH, "results", "lrp_npy_weights")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        weights_dict = {}
        reader = pywrap_tensorflow.NewCheckpointReader(chkp_path)
        name_shape_dict = reader.get_variable_to_shape_map()
        for tensor_name in name_shape_dict.keys():
            if (
                len(name_shape_dict[tensor_name]) == 0
                or "Adam" in tensor_name
                or "opaque_kernel" in tensor_name
            ):
                continue
            splited_name = tensor_name.split("/")
            layer_name = splited_name[1]

            if "conv" in layer_name:
                layer_text_splitted = layer_name.split("_")
                layer_instance_number = layer_text_splitted[1]
                tensor = reader.get_tensor(tensor_name)
                if "bias" in splited_name:
                    np.save(
                        os.path.join(
                            save_folder, "CNN%i-B.npy" % int(layer_instance_number)
                        ),
                        tensor,
                    )
                if "kernel" in splited_name:
                    np.save(
                        os.path.join(
                            save_folder, "CNN%i-W.npy" % int(layer_instance_number)
                        ),
                        tensor,
                    )
                if layer_name not in weights_dict:
                    weights_dict[layer_name] = {}
                weights_dict[layer_name][splited_name[-1]] = reader.get_tensor(
                    tensor_name
                )

            if "dense" in layer_name:
                layer_text_splitted = layer_name.split("_")
                layer_instance_number = layer_text_splitted[1]
                tensor = reader.get_tensor(tensor_name)
                if "bias" in splited_name:
                    np.save(
                        os.path.join(
                            save_folder, "FC%i-B.npy" % int(layer_instance_number)
                        ),
                        tensor,
                    )
                if "kernel" in splited_name:
                    np.save(
                        os.path.join(
                            save_folder, "FC%i-W.npy" % int(layer_instance_number)
                        ),
                        tensor,
                    )
                if layer_name not in weights_dict:
                    weights_dict[layer_name] = {}
                weights_dict[layer_name][splited_name[-1]] = reader.get_tensor(
                    tensor_name
                )

            if "output_logits" in layer_name:
                layer_text_splitted = layer_name.split("_")
                layer_instance_number = layer_text_splitted[1]
                tensor = reader.get_tensor(tensor_name)
                if "bias" in splited_name:
                    np.save(os.path.join(save_folder, "FC3-B.npy"), tensor)
                if "kernel" in splited_name:
                    np.save(os.path.join(save_folder, "FC3-W.npy"), tensor)
                if layer_name not in weights_dict:
                    weights_dict[layer_name] = {}
                weights_dict[layer_name][splited_name[-1]] = reader.get_tensor(
                    tensor_name
                )

        if np.unique(list(weights_dict.keys())).shape[0] != len(
            list(weights_dict.keys())
        ):
            raise ValueError(
                "dimensions mismatch, there is a layer_name appearing more than once"
            )
        return save_folder

    def _load_numpy_weights_to_lrp(self, params_path):
        # values passed by reference
        W, B = self.lrp_net.getWeights()
        weights, biases = self.sess.run([W, B])
        counter_CNN = 1
        counter_FC = 1
        for i in range(len(W)):
            if len(W[i].get_shape().as_list()) == 4:
                modify_op1 = W[i].assign(
                    np.load(params_path + "/CNN" + str(counter_CNN) + "-W.npy")
                )
                modify_op2 = B[i].assign(
                    np.load(params_path + "/CNN" + str(counter_CNN) + "-B.npy")
                )
                self.sess.run((modify_op1, modify_op2))
                counter_CNN += 1
            if len(W[i].get_shape().as_list()) == 2:
                modify_op1 = W[i].assign(
                    np.load(params_path + "/FC" + str(counter_FC) + "-W.npy")
                )
                modify_op2 = B[i].assign(
                    np.load(params_path + "/FC" + str(counter_FC) + "-B.npy")
                )
                self.sess.run((modify_op1, modify_op2))
                counter_FC += 1

    def _init_lrp_visualization_graph(self):
        lrp_method = "alphabeta"
        lrp_method_param = 2
        with tf.name_scope("relevance_propagation"):
            with tf.name_scope("relevance2propagate"):
                # mask for predicted classes LRP
                output_pred_cls_OH_mask = tf.one_hot(
                    indices=self.output_pred_cls,
                    depth=self.params[param_keys.NUMBER_OF_CLASSES],
                )
                # highest logit or model prediction class relevance to propagate up-to the input
                logit2LRP = self.logits * output_pred_cls_OH_mask

                # LRP can be performed over a single logit, the highest on (precited class), true class logit, all logits,
                # or softmax output. LRP over softmax would lose negative logits interpretations, but this statement is not clear.

            with tf.name_scope("LRP"):
                relevance_at_input = self.lrp_net.lrp(
                    logit2LRP, lrp_method, lrp_method_param
                )
                unrotated_relevance = self.unrotate_batch(relevance_at_input)
                average_unrotated_relevance = self.average_rotations(
                    unrotated_relevance
                )
        return (
            logit2LRP,
            relevance_at_input,
            unrotated_relevance,
            average_unrotated_relevance,
        )

    def unrotate_batch(self, batch):
        with tf.name_scope("unrotation"):
            _, h, w, depth = batch.get_shape().as_list()
            # separate batch in 4 diferent rotations
            img_reshape = tf.reshape(batch, [4, -1, h, w, depth])
            # get unrotated images of batch
            img_1 = img_reshape[0, ...]
            img_2 = tf.map_fn(lambda x: tf.image.rot90(x, k=4 - 1), img_reshape[1, ...])
            img_3 = tf.map_fn(lambda x: tf.image.rot90(x, k=4 - 2), img_reshape[2, ...])
            img_4 = tf.map_fn(lambda x: tf.image.rot90(x, k=4 - 3), img_reshape[3, ...])

            unrotated_batch = tf.concat([img_1, img_2, img_3, img_4], 0)
        return unrotated_batch

    def average_rotations(self, batch):
        with tf.name_scope("pool_rotations"):
            _, h, w, depth = batch.get_shape().as_list()
            # separate batch in 4 diferent rotations
            img_reshape = tf.reshape(batch, [4, -1, h, w, depth])

            pooled_img = tf.reduce_mean(img_reshape, [0])
        return pooled_img

    # TODO non by index
    # def plot_relevance_by_index(self, input_images, input_lbl, index):
    #   labels_name = ["bogus", "real"]
    #
    #   # get relevances
    #   self.sess.run(self.iterator_validation.initializer,
    #                 feed_dict={self.validation_img_ph: input_images,
    #                            self.validation_lbl_ph: input_lbl})
    #   mean_relevance, predicted_label, pred_prob, pred_lbl_lrp, pred_prob_lrp = \
    #     self.sess.run(
    #       [self.mean_relevance, self.output_pred_cls, self.output_probabilities,
    #        self.lrp_output_pred_cls, self.lrp_output_probabilities],
    #       feed_dict={self.handle_ph: self.validation_handle,
    #                  self.training_flag: False})
    #
    #   for image_index in range(mean_relevance.shape[0]):
    #     # normalize relevances
    #     normalized_relevances = self.plot_tools.normalize_through_channels(
    #         mean_relevance)
    #     # plot samples
    #     print("Sample: %.0f - Label: %s - Predicted: %s" % (
    #       index, labels_name[input_lbl[index]], labels_name[int(predicted_label)]))
    #     self.plot_tools.plot_input_image(input_images[index, ...])
    #     # plot pooled relevances
    #     print("---Average Relevance---")
    #     self.plot_tools.plot_heatmap(normalized_relevances[0, ...])
    #     # return mean_relevance, predicted_label, normalized_relevances

    def _plot_relevance_by_index(self, input_images, input_lbl, index):
        labels_name = ["AGN", "SN", "VS", "asteroid", "bogus"]

        # get relevances
        self.sess.run(
            self.iterator_validation.initializer,
            feed_dict={
                self.validation_img_ph: input_images[index, ...][np.newaxis, ...],
                self.validation_lbl_ph: [input_lbl[index]],
            },
        )
        (
            mean_relevance,
            predicted_label,
            pred_prob,
            pred_lbl_lrp,
            pred_prob_lrp,
        ) = self.sess.run(
            [
                self.mean_relevance,
                self.output_pred_cls,
                self.output_probabilities,
                self.lrp_output_pred_cls,
                self.lrp_output_probabilities,
            ],
            feed_dict={
                self.handle_ph: self.validation_handle,
                self.training_flag: False,
            },
        )

        # normalize relevances
        normalized_relevances = self.plot_tools.normalize_through_channels(
            mean_relevance
        )
        # plot samples
        # message = "Sample: %.0f - Label: %s - Predicted: %s - LRP_pred_error: %.4f" % (
        #   index, labels_name[input_lbl[index]],
        #   labels_name[int(predicted_label)],
        #   np.abs(np.mean(pred_prob - pred_prob_lrp)))
        # message = "Sample: %.0f - Label: %s - Predicted: %s" % (
        #   index, labels_name[input_lbl[index]],
        #   labels_name[int(predicted_label)])
        message = "Sample: %.0f - Label: %s - Predicted: %s\n Confidence: %.2f" % (
            index,
            labels_name[input_lbl[index]],
            labels_name[int(predicted_label)],
            np.max(pred_prob),
        )
        # print(message)
        self.plot_tools.plot_input_image(input_images[index, ...], sup_title=message)
        # plot pooled relevances
        # print("---Average Relevance---")
        self.plot_tools.plot_heatmap(normalized_relevances[0, ...])
        # return mean_relevance, predicted_label, normalized_relevances

    def plot_relevances(self, imp_im, imp_lb, index=None):
        if index != None:
            self._plot_relevance_by_index(imp_im, imp_lb, index)

        else:
            for i in range(imp_im.shape[0]):
                self._plot_relevance_by_index(imp_im, imp_lb, i)

    def load_model(self, path):
        aux_saver = tf.train.Saver(
            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="network")
        )
        aux_saver.restore(self.sess, path)
        lrp_weights_path = self.save_checkpoint_weights_to_npy(path)
        self._load_numpy_weights_to_lrp(lrp_weights_path)
