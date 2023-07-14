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

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_nans_norm_crop_stamp_model import (
    DeepHiTSNanNormCropStampModel,
)
from modules.data_loaders.frame_to_input import FrameToInput
from parameters import param_keys


# TODO: refactor train test and validate
class DeepHiTSRealBogNanNormCropStampModel(DeepHiTSNanNormCropStampModel):
    """
    Constructor
    """

    def __init__(
        self, params={}, model_name="DeepHitsRealBogNanNormCropStamp", session=None
    ):
        super().__init__(params, model_name, session)

    def _data_loader_init(self, params):
        data_loader = FrameToInput(params)
        data_loader.dataset_preprocessor.set_pipeline(
            [
                data_loader.dataset_preprocessor.image_check_single_image,
                data_loader.dataset_preprocessor.image_clean_misshaped,
                data_loader.dataset_preprocessor.image_select_channels,
                data_loader.dataset_preprocessor.image_normalize_by_image,
                data_loader.dataset_preprocessor.image_nan_to_num,
                data_loader.dataset_preprocessor.image_crop_at_center,
                data_loader.dataset_preprocessor.labels_to_real_bogus,
            ]
        )
        dataset_preprocessor = data_loader.get_dataset_preprocessor()
        return data_loader, dataset_preprocessor
