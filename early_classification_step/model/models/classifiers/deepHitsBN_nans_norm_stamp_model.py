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
from models.classifiers.deepHits_4class_model import DeepHiTS4ClassModel
from modules.data_loaders.frame_to_input import FrameToInput
from modules.networks.deep_hits_bn import DeepHitsBN
from parameters import param_keys, constants


# TODO: refactor train test and validate
class DeepHiTSBNNanNormStampModel(DeepHiTS4ClassModel):
    """
    Constructor
    """

    def __init__(self, params={}, model_name="DeepHitsBNNanNormStamp", session=None):
        super().__init__(params, model_name, session)

    def _update_new_default_params(self):
        new_default_params = {
            param_keys.NUMBER_OF_CLASSES: 4,
            param_keys.KERNEL_SIZE: 3,
            param_keys.BATCHNORM_FC: constants.BN,
            param_keys.BATCHNORM_CONV: constants.BN,
            param_keys.DROP_RATE: 0.5,
            param_keys.BATCH_SIZE: 50,
            param_keys.N_INPUT_CHANNELS: 3,
            param_keys.CHANNELS_TO_USE: [0, 1, 2],
            param_keys.NANS_TO: 0,
        }
        self.params.update(new_default_params)

    def _data_loader_init(self, params):
        data_loader = FrameToInput(params)
        data_loader.dataset_preprocessor.set_pipeline(
            [
                data_loader.dataset_preprocessor.image_check_single_image,
                data_loader.dataset_preprocessor.image_clean_misshaped,
                data_loader.dataset_preprocessor.image_select_channels,
                data_loader.dataset_preprocessor.image_nan_to_num,
            ]
        )
        dataset_preprocessor = data_loader.get_dataset_preprocessor()
        return data_loader, dataset_preprocessor

    def _init_network(self, X, params, training_flag):
        network = DeepHitsBN(X, params, training_flag)
        return network.get_output()
