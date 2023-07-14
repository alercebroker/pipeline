#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 00:17:54 2018
Data splitter by n samples to get
@author: asceta
"""
import warnings
import numpy as np
from modules.data_set_generic import Dataset as data_set_class
from modules.data_set_generic import Dataset
from modules.data_splitters.data_splitter_percentages import DatasetDividerPercentage


# Todo: return dicts instead of separate datasets
class DatasetDividerInt(DatasetDividerPercentage):
    """
    Constructor
    """

    def __init__(
        self,
        data_set_obj=None,
        test_size=100,
        validation_size=100,
        test_random_seed=42,
        val_random_seed=42,
    ):
        super().__init__(
            data_set_obj, test_size, validation_size, test_random_seed, val_random_seed
        )

    def check_size_type(self, value):
        if type(value) != int:
            raise ValueError("set size of value %s is not an int" % str(value))

    def _check_label_values_multiple_of_test_size(self, label_values, test_size):
        if test_size % label_values.shape[0] != 0:
            warnings.warn(
                "test_size %% label_values.shape[0] != 0; set will have size %i"
                % (test_size - (test_size % label_values.shape[0])),
                UserWarning,
                stacklevel=2,
            )

    def _train_test_split(self, data_array, data_label, test_size, random_state):
        label_values = np.unique(data_label)
        self._check_label_values_multiple_of_test_size(label_values, test_size)
        n_labels_per_class = test_size // label_values.shape[0]
        result_dict = {
            "train_array": [],
            "test_array": [],
            "train_labels": [],
            "test_labels": [],
        }
        # print(label_values)
        for single_label_value in label_values:
            class_idxs = np.where(data_label == single_label_value)[0]
            np.random.RandomState(random_state).shuffle(class_idxs)
            # print(class_idxs)
            test_class_idxs = class_idxs[:n_labels_per_class]
            train_class_idxs = class_idxs[n_labels_per_class:]
            result_dict["train_array"].append(data_array[train_class_idxs])
            result_dict["test_array"].append(data_array[test_class_idxs])
            # print(train_class_idxs.shape)
            # print(data_label.shape)
            # print(np.array(data_label)[np.array(train_class_idxs)])
            result_dict["train_labels"].append(data_label[train_class_idxs])
            result_dict["test_labels"].append(data_label[test_class_idxs])

        return (
            np.concatenate(result_dict["train_array"]),
            np.concatenate(result_dict["test_array"]),
            np.concatenate(result_dict["train_labels"]),
            np.concatenate(result_dict["test_labels"]),
        )

    def get_train_test_val_set_objs(self):
        train_data_set, test_data_set = self.get_train_test_data_set_objs()
        train_data_set, validation_data_set = self._split_data_in_sets(
            train_data_set, self.validation_size, self.val_random_state
        )
        return train_data_set, test_data_set, validation_data_set
