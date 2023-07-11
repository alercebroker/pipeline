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
from modules.iterators import train_iterator, validation_iterator
from parameters import param_keys
from models.classifiers.base_model import BaseModel
from modules.data_loaders.ztf_stamps_loader import ZTFLoader
from modules.iterators.iterator_pre_processing import ZTFPreprocessorTf


# TODO: refactor train test and validate
class BaseModelPreprocessingInPipeline(BaseModel):
    """
    Constructor
    """

    def __init__(
        self, params={}, model_name="BasePreprocessingInPipeline", session=None
    ):
        super().__init__(params, model_name, session)

    def _data_loader_init(self, params):
        data_loader = ZTFLoader(params)
        data_loader.dataset_preprocessor.set_pipeline(
            [
                data_loader.dataset_preprocessor.image_check_single_image,
                data_loader.dataset_preprocessor.image_clean_misshaped,
            ]
        )
        dataset_preprocessor = data_loader.get_dataset_preprocessor()
        return data_loader, dataset_preprocessor

    def _tf_preprocessor_init(self, params):
        iterator_preprocessor = ZTFPreprocessorTf(params)
        return iterator_preprocessor

    # TODO implement builder pattern to avoid code replication and reduce 2 lines
    def _iterator_init(self, params):
        tf_preprocessor = self._tf_preprocessor_init(params)
        with tf.name_scope("iterators"):
            train_it_builder = train_iterator.TrainIteratorBuilder(
                params, pre_batch_processing=tf_preprocessor.preprocess_dataset
            )
            (
                iterator_train,
                train_sample_ph,
                train_lbl_ph,
            ) = train_it_builder.get_iterator_and_ph()
            val_it_builder = validation_iterator.ValidationIteratorBuilder(
                params, pre_batch_processing=tf_preprocessor.preprocess_dataset
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
    params = {
        param_keys.CHANNELS_TO_USE: 0,
        param_keys.N_INPUT_CHANNELS: 1,
    }
    model = BaseModelPreprocessingInPipeline(params=params)
    model.create_paths()
    train_writer = tf.summary.FileWriter(
        os.path.join(model.tb_path, "train"), model.sess.graph
    )
    model.close()
