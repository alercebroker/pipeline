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
from models.classifiers.deepHits_nans_norm_crop_stamp_model import (
    DeepHiTSNanNormCropStampModel,
)
from modules.data_loaders.frame_to_input import FrameToInput
from modules.networks.wide_net import WideNet
from parameters import param_keys, general_keys

# TODO: refactor train test and validate
class WideNetModel(DeepHiTSNanNormCropStampModel):
    """
    Constructor
    """

    def __init__(self, params={}, model_name="WideNet", session=None):
        super().__init__(params, model_name, session)

    def _init_network(self, X, params, training_flag):
        network = WideNet(X, params, training_flag)
        return network.get_output()

    def _data_loader_init(self, params):
        data_loader = FrameToInput(params)
        data_loader.dataset_preprocessor.set_pipeline(
            [
                data_loader.dataset_preprocessor.image_check_single_image,
                data_loader.dataset_preprocessor.image_clean_misshaped,
                data_loader.dataset_preprocessor.image_select_channels,
                data_loader.dataset_preprocessor.image_normalize_by_image,
                data_loader.dataset_preprocessor.image_nan_to_num,
            ]
        )
        dataset_preprocessor = data_loader.get_dataset_preprocessor()
        return data_loader, dataset_preprocessor


if __name__ == "__main__":
    n_classes = 5
    params = {
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TRAIN_ITERATIONS_HORIZON: 30000,
        param_keys.TRAIN_HORIZON_INCREMENT: 10000,
        param_keys.TEST_SIZE: n_classes * 50,
        param_keys.VAL_SIZE: n_classes * 50,
        param_keys.NANS_TO: 0,
        param_keys.NUMBER_OF_CLASSES: n_classes,
        param_keys.CROP_SIZE: 21,
        param_keys.INPUT_IMAGE_SIZE: 21,
        param_keys.VALIDATION_MONITOR: general_keys.LOSS,
        param_keys.VALIDATION_MODE: general_keys.MIN,
        param_keys.ENTROPY_REG_BETA: None,
    }
    model = WideNetModel(params)
    model.create_paths()
    train_writer = tf.summary.FileWriter(
        os.path.join(model.tb_path, "train"), model.sess.graph
    )
    model.close()
