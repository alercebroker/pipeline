#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 00:17:54 2018

Data splitter by percentages

@author: asceta
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import numpy as np
from sklearn.model_selection import train_test_split

from modules.data_set_generic import Dataset
from modules.data_set_generic import Dataset as data_set_class


# Todo: return dicts instead of separate datasets
class DatasetDividerPercentage(object):
    """
    Constructor
    """

    def __init__(
        self,
        data_set_obj=None,
        test_size=0.5,
        validation_size=0.1,
        test_random_seed=42,
        val_random_seed=42,
    ):
        self.data_set_obj = data_set_obj
        if data_set_obj:  # is not None
            self.batch_size = self.data_set_obj.batch_size
        self.test_size = test_size
        self.validation_size = validation_size
        self.test_random_state = test_random_seed
        self.val_random_state = val_random_seed
        self.check_size_type(test_size)
        self.check_size_type(validation_size)

    def check_size_type(self, value):
        if type(value) != float:
            raise ValueError("set size of value %s is not a float" % str(value))

    def set_dataset_obj(self, dataset_obj):
        self.check_if_dataset_contains_labels(dataset_obj)
        self.data_set_obj = dataset_obj
        self.batch_size = self.data_set_obj.batch_size

    def check_if_dataset_contains_labels(self, dataset: Dataset):
        unique_labels = np.unique(dataset.data_label)
        if len(unique_labels) == 1:
            warnings.warn(
                "Dataset object to be splitted has ONLY one type of label: %s."
                " Split WON'T be label stratified!" % str(unique_labels[0]),
                UserWarning,
            )

    def _train_test_split(self, data_array, data_label, test_size, random_state):
        train_test_split(
            data_array,
            data_label,
            test_size=test_size,
            random_state=random_state,
            stratify=data_label,
        )

    def _split_data_in_sets(self, data_set_obj, size, random_seed):
        self.check_size_type(size)
        train_idxs, test_idxs, _, _ = self._train_test_split(
            np.arange(data_set_obj.data_array.shape[0]),
            data_set_obj.data_label,
            test_size=size,
            random_state=random_seed,
        )
        train_data_set = data_set_class(
            data_set_obj.data_array[train_idxs],
            data_set_obj.data_label[train_idxs],
            meta_data=data_set_obj.meta_data[train_idxs],
            batch_size=self.batch_size,
        )
        test_data_set = data_set_class(
            data_set_obj.data_array[test_idxs],
            data_set_obj.data_label[test_idxs],
            meta_data=data_set_obj.meta_data[test_idxs],
            batch_size=self.batch_size,
        )
        return train_data_set, test_data_set

    def get_train_test_val_set_objs(self):
        train_data_set, test_data_set = self.get_train_test_data_set_objs()
        train_data_set, validation_data_set = self._split_data_in_sets(
            train_data_set,
            self.validation_size / (1 - self.test_size),
            self.val_random_state,
        )
        return train_data_set, test_data_set, validation_data_set

    def get_train_test_data_set_objs(self):
        return self._split_data_in_sets(
            self.data_set_obj, self.test_size, self.test_random_state
        )

    def get_train_val_data_set_objs(self):
        return self._split_data_in_sets(
            self.data_set_obj, self.validation_size, self.val_random_state
        )
