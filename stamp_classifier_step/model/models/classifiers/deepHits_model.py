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
import tensorflow as tf

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)
from modules.networks.deep_hits import DeepHits
from modules.networks.deep_hits_extra_layer import DeepHitsExtraLayer
from modules.iterators import train_iterator, validation_iterator
from parameters import param_keys, general_keys
from models.classifiers.base_model import BaseModel
from modules.iterators.iterator_post_processing import augment_with_rotations


# TODO: refactor train test and validate
class DeepHitsModel(BaseModel):
    """
    Constructor
    """

    def __init__(self, params={}, model_name="DeepHits", session=None):
        super().__init__(params, model_name, session)

    def _update_new_default_params(self):
        new_default_params = {
            param_keys.KERNEL_SIZE: 3,
            param_keys.BATCHNORM_FC: None,
            param_keys.BATCHNORM_CONV: None,
            param_keys.DROP_RATE: 0.5,
            param_keys.BATCH_SIZE: 50,
        }
        self.params.update(new_default_params)

    def _init_network(self, X, params, training_flag):
        network = DeepHitsExtraLayer(X, params, training_flag)
        return network.get_output()

    # TODO implement builder pattern to avoid code replication and reduce 2 lines
    def _iterator_init(self, params):
        with tf.name_scope("iterators"):
            train_it_builder = train_iterator.TrainIteratorBuilder(
                params, post_batch_processing=augment_with_rotations
            )
            (
                iterator_train,
                train_sample_ph,
                train_lbl_ph,
            ) = train_it_builder.get_iterator_and_ph()
            val_it_builder = validation_iterator.ValidationIteratorBuilder(
                params, post_batch_processing=augment_with_rotations
            )
            (
                iterator_val,
                val_sample_ph,
                val_lbl_ph,
            ) = val_it_builder.get_iterator_and_ph()
            handle_ph, global_iterator = train_it_builder.get_global_iterator()
        return (
            global_iterator,
            handle_ph,
            train_sample_ph,
            train_lbl_ph,
            iterator_train,
            val_sample_ph,
            val_lbl_ph,
            iterator_val,
        )


if __name__ == "__main__":
    model = DeepHitsModel()
    model.create_paths()
    train_writer = tf.summary.FileWriter(
        os.path.join(model.tb_path, "train"), model.sess.graph
    )
    model.close()
