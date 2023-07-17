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
from models.classifiers.base_model import BaseModel
from modules.data_set_generic import Dataset
from modules.data_splitters.data_splitter_n_samples import DatasetDividerInt
from modules.data_loaders.frame_to_input import FrameToInput
from parameters import param_keys, general_keys


# TODO: refactor train test and validate
class Base4ClassModel(BaseModel):
    """
    Constructor
    """

    def __init__(self, params={}, model_name="Base4class", session=None):
        super().__init__(params, model_name, session)

    def _update_new_default_params(self):
        new_default_params = {
            param_keys.NUMBER_OF_CLASSES: 4,
        }
        self.params.update(new_default_params)

    def _prepare_input(self, X, y, validation_data, test_data):
        if X.shape != () and y.shape != () and validation_data:
            train_set = Dataset(X, y, self.params[param_keys.BATCH_SIZE])
            val_set = Dataset(
                validation_data[0],
                validation_data[1],
                self.params[param_keys.BATCH_SIZE],
            )
            test_set = self._check_test_data_availability(test_data)

        elif X.shape != () and y.shape != () and not validation_data:
            aux_dataset_obj = Dataset(X, y, self.params[param_keys.BATCH_SIZE])
            data_divider = DatasetDividerInt(
                aux_dataset_obj,
                validation_size=self.params[param_keys.VAL_SIZE],
                val_random_seed=self.params[param_keys.VALIDATION_RANDOM_SEED],
            )
            train_set, val_set = data_divider.get_train_val_data_set_objs()
            test_set = self._check_test_data_availability(test_data)

        else:  # valdiation_data and test_data ignored
            train_set, val_set, test_set = self._data_init()
        train_set.balance_data_by_replication()
        train_set = self._global_shuffling(train_set)
        return train_set, val_set, test_set

    # # TODO: change return for dict
    # def _data_init(self):
    #   data_loader = FrameToInput(self.params)
    #   darasets_dict = data_loader.get_datasets()
    #   return darasets_dict[general_keys.TRAIN], \
    #          darasets_dict[general_keys.VALIDATION], \
    #          darasets_dict[general_keys.TEST]

    def _data_loader_init(self, params):
        data_loader = FrameToInput(params)
        dataset_preprocessor = data_loader.get_dataset_preprocessor()
        return data_loader, dataset_preprocessor


if __name__ == "__main__":
    model = Base4ClassModel()
    model.create_paths()
    train_writer = tf.summary.FileWriter(
        os.path.join(model.tb_path, "train"), model.sess.graph
    )
    model.close()
