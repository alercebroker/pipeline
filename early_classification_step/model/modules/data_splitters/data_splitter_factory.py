#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 00:17:54 2018

ZTF Dataset

@author: asceta
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.model_selection import train_test_split

from modules.data_set_generic import Dataset as data_set_class

# Todo: CODE
class DatasetDivider(object):

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

    def set_dataset_obj(self, dataset_obj):
        self.data_set_obj = dataset_obj
        self.batch_size = self.data_set_obj.batch_size

    def _split_data_in_sets(self, data_set_obj, size, random_seed):
        train_array, test_array, train_labels, test_labels = train_test_split(
            data_set_obj.data_array,
            data_set_obj.data_label,
            test_size=size,
            random_state=random_seed,
        )  # ,
        # stratify=data_set_obj.data_label)
        train_data_set = data_set_class(train_array, train_labels, self.batch_size)
        test_data_set = data_set_class(test_array, test_labels, self.batch_size)
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
