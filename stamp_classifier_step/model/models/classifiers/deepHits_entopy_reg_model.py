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
from parameters import param_keys
import modules.losses as losses


# TODO: refactor train test and validate
class DeepHiTSEntropyRegModel(DeepHiTSNanNormCropStampModel):
    """
    Constructor
    """

    def __init__(self, params={}, model_name="DeepHitsEntropyReg", session=None):
        super().__init__(params, model_name, session)

    def _update_new_default_params(self):
        new_default_params = {
            param_keys.NUMBER_OF_CLASSES: None,
            param_keys.KERNEL_SIZE: 3,
            param_keys.BATCHNORM_FC: None,
            param_keys.BATCHNORM_CONV: None,
            param_keys.DROP_RATE: 0.5,
            param_keys.BATCH_SIZE: 50,
            param_keys.N_INPUT_CHANNELS: 3,
            param_keys.CHANNELS_TO_USE: [0, 1, 2],
            param_keys.NANS_TO: 0,
            param_keys.INPUT_IMAGE_SIZE: 21,
            param_keys.CROP_SIZE: 21,
            param_keys.ENTROPY_REG_BETA: 1e-3,
        }
        self.params.update(new_default_params)

    def _loss_init(self, logits, input_labels, number_of_classes):
        with tf.name_scope("loss_function"):
            loss = losses.xentropy_with_entropy_regularization(
                logits,
                input_labels,
                number_of_classes,
                self.params[param_keys.ENTROPY_REG_BETA],
            )
        return loss
