"""Module that defines common errors for parameter values."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from . import constants
import pandas as pd


def check_valid_value(value, name, valid_list):
    """Raises a ValueError exception if value not in valid_list"""
    if value not in valid_list:
        msg = constants.ERROR_INVALID % (valid_list, name, value)
        raise ValueError(msg)


def check_nan_metric(metric_dict, metric, metric_mean):
    """Raises a ValueError exception if metric value is nan"""
    for (
        name,
        tensor,
    ) in (
        metric_dict.items()
    ):  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if tensor == metric:
            metric_name = name
    if np.isnan(metric_mean):
        msg = constants.ERROR_NAN_METRIC % (metric_name)
        raise ValueError(msg)


def check_data_frama_contain_features(
    data_frame: pd.DataFrame, features_names_list: list
):
    """Raises a ValueError exception if data_frame don't contain any of wanted
    features"""
    df_headers = list(data_frame.columns.values)
    if not all(feature_name in data_frame for feature_name in features_names_list):
        missing_features_idxs = [
            feature_name not in data_frame for feature_name in features_names_list
        ]
        missing_features = np.array(features_names_list)[missing_features_idxs]
        msg = constants.ERROR_DATAFRAME_FEATURES % str(missing_features)
        raise ValueError(msg)
