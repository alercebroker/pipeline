#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing ZTF database to be saved as a samplesx21x21x3 numpy array in a pickle 

TODO: clean_NaN once cropped
TODO: unit tests
ToDo: instead of cascade implement as pipeline, in order to have single call and definition
ToDo: smart way to shut down nans
@author: asceta
"""
import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)

# from modules.data_set_alerce import DatasetAlerce as Dataset
from modules.data_set_generic import Dataset as DatasetObj
import numpy as np
from modules.data_loaders.ztf_preprocessor import ZTFDataPreprocessor
from parameters import general_keys
import numbers
import warnings
from parameters import param_keys


# Todo: refactor verbose
# comopse as a pipeline to choose preprocessing steps
class ATLASDataPreprocessor(ZTFDataPreprocessor):
    """
    Constructor
    """

    def __init__(self, params, verbose=True):
        super().__init__(params, verbose)

    def labels_to_kast_streaks_artifact(self, dataset: DatasetObj):
        unique_labels = np.unique(dataset.data_label)
        if len(unique_labels) == 3:
            return dataset
        artifact_label_values = [0, 2, 3, 5, 6]
        kast_label_value = 4
        streak_label_value = 1
        final_labels = np.zeros_like(dataset.data_label)
        kast_indexes = np.where(dataset.data_label == kast_label_value)[0]
        streak_indexes = np.where(dataset.data_label == streak_label_value)[0]
        final_labels[kast_indexes] = 1
        final_labels[streak_indexes] = 2
        random_idxs = np.random.randint(0, len(final_labels), 100)
        # print(final_labels[random_idxs])
        # print(dataset.data_label[random_idxs])
        dataset.data_label = final_labels
        if self.verbose:
            print(
                "Labels changed to Artifact - Kast - Straks\n%s"
                % (
                    self._get_string_label_count(
                        dataset.data_label, np.array(["artifact", "kast", "streak"])
                    )
                ),
                flush=True,
            )
        return dataset

    def remove_streaks(self, dataset: DatasetObj):
        if len(np.unique(dataset.data_label)) == 7:
            streak_label_value = 1
            dataset.data_array = dataset.data_array[
                dataset.data_label != streak_label_value
            ]
            dataset.meta_data = dataset.meta_data[
                dataset.data_label != streak_label_value
            ]
            dataset.data_label = dataset.data_label[
                dataset.data_label != streak_label_value
            ]
            dataset.data_label[dataset.data_label > streak_label_value] = (
                dataset.data_label[dataset.data_label > streak_label_value] - 1
            )
        if self.verbose:
            print(
                "Streaks removed\n%s"
                % (
                    self._get_string_label_count(
                        dataset.data_label,
                        np.array(["cr", "burn", "scar", "kast", "spike", "noise"]),
                    )
                ),
                flush=True,
            )
        return dataset

    def images_to_gray_scale(self, dataset: DatasetObj):
        if dataset.data_array.shape[-1] == 1:
            return dataset
        rgb = dataset.data_array
        gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])[..., None]
        # print(gray.shape)
        dataset.data_array = gray
        return dataset

    def _init_feature_normalization_stats_dict(self, dataset: DatasetObj):
        print(
            "Calculating feature statistics for normalization,"
            " with set of size %s" % str(dataset.meta_data.shape)
        )
        self.feature_normalization_stats_dict = {}
        features_names_list = self.params[param_keys.FEATURES_NAMES_LIST]
        for feature_name in features_names_list:
            feature_values = dataset.meta_data[:, feature_name]
            if isinstance(feature_values[0], numbers.Number) == False:
                continue
            self.feature_normalization_stats_dict[feature_name] = {
                general_keys.MEAN: np.mean(feature_values),
                general_keys.STD: np.std(feature_values),
            }
        print(
            "Using %i features"
            % len(list(self.feature_normalization_stats_dict.keys()))
        )
        # print(self.feature_normalization_stats_dict)

    def features_normalize(self, dataset: DatasetObj):
        if self.use_feature_normalizer == False:
            if self.verbose_warnings:
                warnings.warn(
                    "Feature normalization is OFF on set of size %i. "
                    "To turn it ON use .set_use_feature_normalization(True)"
                    % len(dataset.data_label),
                    UserWarning,
                )
            return dataset
        features_names_list = self.params[param_keys.FEATURES_NAMES_LIST]
        if self.feature_normalization_stats_dict is None:
            self._init_feature_normalization_stats_dict(dataset)
        metadata_normed = np.empty(
            [dataset.meta_data.shape[0], len(features_names_list)]
        )
        for i, feature_name in enumerate(features_names_list):
            if feature_name not in self.feature_normalization_stats_dict:
                continue
            feature_values = dataset.meta_data[:, feature_name]
            features_mean = self.feature_normalization_stats_dict[feature_name][
                general_keys.MEAN
            ]
            features_std = self.feature_normalization_stats_dict[feature_name][
                general_keys.STD
            ]
            if features_std == 0:
                features_std = 1
            norm_feature_values = (feature_values - features_mean) / features_std
            metadata_normed[:, i] = norm_feature_values
        dataset.meta_data = metadata_normed
        return dataset

    def _get_misshaped_samples_idx(self, samples):
        miss_shaped_sample_idx = []
        for i in range(len(samples)):
            sample = samples[i]
            if sample.shape[0] == sample.shape[1] and sample.shape[0] == self.crop_size:
                continue
            if sample.shape[1] != 101 or sample.shape[0] != 101:
                # print("sample %i of shape %s" % (i, str(sample.shape)))
                miss_shaped_sample_idx.append(i)
        self._check_misshape_all_removed(samples, miss_shaped_sample_idx)
        return miss_shaped_sample_idx

    def _get_nans_metadata_samples_idx(self, metadata_samples):
        nans_sample_idx = []
        for i in range(len(metadata_samples)):
            sample = metadata_samples[i]
            if (~np.isfinite(sample)).any():
                # print("sample %i of shape %s" %(i,str(sample.shape)))
                nans_sample_idx.append(i)
        return nans_sample_idx

    def metadata_nan_to_num(self, dataset: DatasetObj):
        samples = dataset.meta_data
        nans_sample_idx = self._get_nans_metadata_samples_idx(samples)
        if self.verbose:
            print(
                "%i metadata samples with NaNs or Inf. NaNs and Inf replaced with number %s"
                % (len(nans_sample_idx), str(self.number_to_replace_nans))
            )
        samples[~np.isfinite(samples)] = self.number_to_replace_nans
        dataset.meta_data = samples
        return dataset
