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

from parameters import param_keys, general_keys

# from modules.data_set_alerce import DatasetAlerce as Dataset
from modules.data_set_generic import Dataset as DatasetObj
import numpy as np
import numbers
import warnings


# Todo: refactor verbose
# comopse as a pipeline to choose preprocessing steps
class ZTFDataPreprocessor(object):
    """
    Constructor
    """

    def __init__(self, params, verbose=True):
        self.params = params
        self.channels_to_select = params[param_keys.CHANNELS_TO_USE]
        self.number_to_replace_nans = params[param_keys.NANS_TO]
        self.crop_size = params[param_keys.CROP_SIZE]
        self.preprocessing_pipeline = [self.identity]
        self.verbose = verbose
        self.verbose_warnings = True
        self.use_feature_normalizer = True
        self.feature_normalization_stats_dict = None

    def set_use_of_feature_normalizer(self, use_feature_normalizer):
        self.use_feature_normalizer = use_feature_normalizer

    """
  define your preprocessing strategy here
  """

    def preprocess_dataset(self, dataset: DatasetObj):
        if self.verbose:
            print("%s" % self._get_string_label_count(dataset.data_label), flush=True)
        for preprocessing_function in self.preprocessing_pipeline:
            dataset = preprocessing_function(dataset)
        # self.verbose = False
        return dataset

    def append_to_pipeline(self, method):
        self.preprocessing_pipeline.append(method)
        return self

    def set_pipeline(self, pipeline):
        self.preprocessing_pipeline = pipeline

    def identity(self, dataset: DatasetObj):
        return dataset

    def image_check_single_image(self, dataset: DatasetObj):
        if len(dataset.data_array.shape) == 3:
            dataset.data_array = dataset.data_array[np.newaxis, ...]
        return dataset

    # TODO: erase single image check; adding dummy at begining
    def image_select_channels(self, dataset: DatasetObj):
        if len(dataset.data_array.shape) == 3:
            dataset.data_array = dataset.data_array[np.newaxis, ...]
        selected_images_channels = dataset.data_array[..., self.channels_to_select]
        if len(selected_images_channels.shape) == 3:
            selected_images_channels = selected_images_channels[..., np.newaxis]
        dataset.data_array = selected_images_channels
        return dataset

    # TODO: normalize template to avoid replication with by_image
    def image_normalize_by_sample(self, dataset: DatasetObj):
        images = dataset.data_array
        images -= np.nanmin(images, axis=(1, 2, 3))[
            ..., np.newaxis, np.newaxis, np.newaxis
        ]
        images = (
            images
            / np.nanmax(images, axis=(1, 2, 3))[..., np.newaxis, np.newaxis, np.newaxis]
        )
        dataset.data_array = images
        return dataset

    def image_normalize_by_image(self, dataset: DatasetObj):
        images = dataset.data_array
        images -= np.nanmin(images, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        images = images / np.nanmax(images, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        dataset.data_array = images
        return dataset

    def image_normalize_by_image_1_1(self, dataset: DatasetObj):
        images = dataset.data_array
        images -= np.nanmin(images, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        images = images / np.nanmax(images, axis=(1, 2))[:, np.newaxis, np.newaxis, :]
        images = (images * 2) - 1
        dataset.data_array = images
        return dataset

    def image_nan_to_num(self, dataset: DatasetObj):
        samples = dataset.data_array
        nans_sample_idx = self._get_nans_samples_idx(samples)
        if self.verbose:
            print(
                "%i samples with NaNs. NaNs replaced with number %s"
                % (len(nans_sample_idx), str(self.number_to_replace_nans))
            )
        samples[np.isnan(samples)] = self.number_to_replace_nans
        dataset.data_array = samples
        return dataset

    def _check_all_removed(self, remove_name, samples_list, idxs_to_remove):
        if len(samples_list) == len(idxs_to_remove):
            raise OverflowError(
                "All samples have %s, thus batch is empty and cannot be processed"
                % remove_name
            )

    def _check_misshape_all_removed(self, samples_list, idxs_to_remove):
        self._check_all_removed("MISSHAPE", samples_list, idxs_to_remove)

    def _check_nan_all_removed(self, samples_list, idxs_to_remove):
        self._check_all_removed("NAN", samples_list, idxs_to_remove)

    def _get_misshaped_samples_idx(self, samples):
        miss_shaped_sample_idx = []
        for i in range(len(samples)):
            sample = samples[i]
            if sample.shape[0] == sample.shape[1] and sample.shape[0] == self.crop_size:
                continue
            if sample.shape[2] != 3 or sample.shape[1] != 63 or sample.shape[0] != 63:
                # if sample.shape[2] != 3 or sample.shape[1] != sample.shape[0]:
                # print("sample %i of shape %s" % (i, str(sample.shape)))
                miss_shaped_sample_idx.append(i)
        self._check_misshape_all_removed(samples, miss_shaped_sample_idx)
        return miss_shaped_sample_idx

    def image_clean_misshaped(self, dataset: DatasetObj):
        samples_clone = list(dataset.data_array[:])
        labels_clone = list(dataset.data_label[:])
        metadata_clone = list(dataset.meta_data[:])
        miss_shaped_sample_idx = self._get_misshaped_samples_idx(samples_clone)
        for index in sorted(miss_shaped_sample_idx, reverse=True):
            samples_clone.pop(index)
            labels_clone.pop(index)
            metadata_clone.pop(index)
        if self.verbose:
            print(
                "%i misshaped samples removed\n%s"
                % (
                    len(miss_shaped_sample_idx),
                    self._get_string_label_count(labels_clone),
                ),
                flush=True,
            )
        dataset = DatasetObj(
            data_array=samples_clone,
            data_label=labels_clone,
            meta_data=metadata_clone,
            batch_size=dataset.batch_size,
        )
        return dataset

    def _get_nans_samples_idx(self, samples):
        nans_sample_idx = []
        for i in range(len(samples)):
            sample = samples[i]
            if np.isnan(sample).any():
                # print("sample %i of shape %s" %(i,str(sample.shape)))
                nans_sample_idx.append(i)
        return nans_sample_idx

    # TODO: refactor; fuse with clean misshaped
    def image_clean_nans(self, dataset: DatasetObj):
        samples_clone = list(dataset.data_array[:])
        labels_clone = list(dataset.data_label[:])
        metadata_clone = list(dataset.meta_data[:])
        nans_sample_idx = self._get_nans_samples_idx(samples_clone)
        self._check_nan_all_removed(samples_clone, nans_sample_idx)
        for index in sorted(nans_sample_idx, reverse=True):
            samples_clone.pop(index)
            labels_clone.pop(index)
            metadata_clone.pop(index)
        if self.verbose:
            print(
                "%i samples with NaNs removed\n%s"
                % (len(nans_sample_idx), self._get_string_label_count(labels_clone)),
                flush=True,
            )
        dataset = DatasetObj(
            data_array=samples_clone,
            data_label=labels_clone,
            batch_size=dataset.batch_size,
            meta_data=metadata_clone,
        )
        return dataset

    def image_crop_at_center(self, dataset: DatasetObj):
        if self.crop_size is None:
            return dataset
        samples = dataset.data_array
        assert samples.shape[1] % 2 == self.crop_size % 2
        center = int((samples.shape[1]) / 2)
        crop_side = int(self.crop_size / 2)
        crop_begin = center - crop_side
        if samples.shape[1] % 2 == 0:
            crop_end = center + crop_side
        elif samples.shape[1] % 2 == 1:
            crop_end = center + crop_side + 1
        # print(center)
        # print(crop_begin, crop_end)
        cropped_samples = samples[:, crop_begin:crop_end, crop_begin:crop_end, :]
        dataset.data_array = cropped_samples
        return dataset

    def _get_string_label_count(
        self, labels, class_names=np.array(["AGN", "SN", "VS", "asteroid", "bogus"])
    ):
        label_values, label_counts = np.unique(labels, return_counts=True)
        if len(label_values) != class_names.shape[0]:
            return ""
        count_dict = dict(zip(label_values, label_counts))
        return_str = "Label count "
        for single_label_value in count_dict.keys():
            return_str += "%s: %i -" % (
                class_names[single_label_value],
                count_dict[single_label_value],
            )
        return return_str

    def labels_to_real_bogus(self, dataset: DatasetObj):
        bogus_label_value = self.params[param_keys.BOGUS_LABEL_VALUE]
        if bogus_label_value is None:
            label_values = np.unique(dataset.data_label)
            bogus_label_value = label_values[-1]
        bogus_indexes = np.where(dataset.data_label == bogus_label_value)[0]
        real_indexes = np.where(dataset.data_label != bogus_label_value)[0]
        dataset.data_label[bogus_indexes] = 0
        dataset.data_label[real_indexes] = 1
        if self.verbose:
            print(
                "Labels changed to Real - Bogus\n%s"
                % (
                    self._get_string_label_count(
                        dataset.data_label, np.array(["bogus", "real"])
                    )
                ),
                flush=True,
            )
        return dataset

    def _noneify_max_min_in_features_clipping_dict(
        self, feature_clipping_dict: dict
    ) -> dict:
        for key in feature_clipping_dict.keys():
            feature_clipping_dict[key] = [
                x if isinstance(x, numbers.Number) else None
                for x in feature_clipping_dict[key]
            ]

    def features_clip(self, dataset: DatasetObj):
        features_names_list = self.params[param_keys.FEATURES_NAMES_LIST]
        features_cliping_dict = self.params[param_keys.FEATURES_CLIPPING_DICT]
        self._noneify_max_min_in_features_clipping_dict(features_cliping_dict)
        for feature_name_idx, feature_name in enumerate(features_names_list):
            if feature_name not in features_cliping_dict:
                continue
            clip_values = features_cliping_dict[feature_name]
            np.clip(
                dataset.meta_data[:, feature_name_idx],
                clip_values[0],
                clip_values[1],
                out=dataset.meta_data[:, feature_name_idx],
            )
        return dataset

    def _init_feature_normalization_stats_dict(self, dataset: DatasetObj):
        print(
            "Calculating feature statistics for normalization,"
            " with set of size %s" % str(dataset.meta_data.shape)
        )
        self.feature_normalization_stats_dict = {}
        features_names_list = self.params[param_keys.FEATURES_NAMES_LIST]
        for feature_name_idx, feature_name in enumerate(features_names_list):
            feature_values = dataset.meta_data[:, feature_name_idx]
            if isinstance(feature_values[0], numbers.Number) == False:
                continue
            self.feature_normalization_stats_dict[feature_name] = {
                general_keys.MEAN: np.mean(feature_values),
                general_keys.STD: np.std(feature_values),
            }

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
        for feature_name_idx, feature_name in enumerate(features_names_list):
            if feature_name not in self.feature_normalization_stats_dict:
                continue
            feature_values = dataset.meta_data[:, feature_name_idx]
            features_mean = self.feature_normalization_stats_dict[feature_name][
                general_keys.MEAN
            ]
            features_std = self.feature_normalization_stats_dict[feature_name][
                general_keys.STD
            ]
            norm_feature_values = (feature_values - features_mean) / features_std
            dataset.meta_data[:, feature_name_idx] = norm_feature_values
        return dataset


if __name__ == "__main__":
    import pandas as pd

    params = {
        param_keys.DATA_PATH_TRAIN: "/home/rcarrasco/stamp_classifier/pickles/alerts_datadict.pkl",
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.CROP_SIZE: 21,
        param_keys.NANS_TO: 0,
    }
    data_dict = pd.read_pickle(params[param_keys.DATA_PATH_TRAIN])
    print(data_dict["labels"])
    print(data_dict["images"][:5])
    print(data_dict["images"][0].shape)
    print(np.unique(data_dict["labels"], return_counts=True))
    dataset_object = DatasetObj(
        data_array=data_dict["images"], data_label=data_dict["labels"]
    )
    dataset_preprocessor = ZTFDataPreprocessor(params)
    dataset_preprocessor.set_pipeline(
        [
            dataset_preprocessor.image_check_single_image,
            dataset_preprocessor.image_clean_misshaped,
            dataset_preprocessor.image_select_channels,
            dataset_preprocessor.image_crop_at_center,
            dataset_preprocessor.image_normalize_by_image,
            dataset_preprocessor.image_nan_to_num,
        ]
    )
    preprocessed_dataset = dataset_preprocessor.preprocess_dataset(dataset_object)
