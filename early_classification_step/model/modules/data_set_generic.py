#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 18:05:40 2018
Dataset Object
Nit everything ready to handle features, but replication is
@author: ereyes
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np


# Todo: refactor metadata
class Dataset(object):
    """
    Constructor
    """

    def __init__(self, data_array, data_label, batch_size, meta_data=None):
        self.batch_counter = 0
        self.batch_counter_val = 0
        self.batch_size = batch_size
        self.data_array = np.array(data_array)
        self.data_label = np.array(
            self._check_data_avaliablility_or_fill_with_zeros(data_label)
        )
        self.meta_data = np.array(
            self._check_data_avaliablility_or_fill_with_zeros(meta_data)
        )

    def _check_data_avaliablility_or_fill_with_zeros(self, data_labels):
        if data_labels is None:
            return np.ones(self.data_array.shape[0]) * -999
        return data_labels

    def _oversampling_array(self, data_array):
        max_label_count, _ = self.get_max_min_label_count()
        n_copies = max_label_count // len(data_array)
        augmented_samples = []
        for i in range(n_copies):
            augmented_samples.append(copy.deepcopy(data_array))
        n_extra = max_label_count - len(np.concatenate(augmented_samples))
        # n_extra = max_label_count - len(augmented_samples)
        augmented_samples.append(copy.deepcopy(data_array[:n_extra]))
        return np.concatenate(augmented_samples)

    def get_batch_images(self):
        batch, _, _ = self.get_batch()

        return batch

    def get_batch(self):
        if self.batch_counter + self.batch_size < self.data_array.shape[0]:
            batch_image = self.data_array[
                self.batch_counter : self.batch_counter + self.batch_size, ...
            ]
            batch_label = self.data_label[
                self.batch_counter : self.batch_counter + self.batch_size, ...
            ]
            batch_metadata = self.meta_data[
                self.batch_counter : self.batch_counter + self.batch_size, ...
            ]
            self.batch_counter += self.batch_size
            # print(get_batch.BATCH_COUNTER)
        else:
            self.batch_counter = 0
            self.shuffle_data()
            batch_image = self.data_array[
                self.batch_counter : self.batch_counter + self.batch_size, ...
            ]
            batch_label = self.data_label[
                self.batch_counter : self.batch_counter + self.batch_size, ...
            ]
            batch_metadata = self.meta_data[
                self.batch_counter : self.batch_counter + self.batch_size, ...
            ]
            self.batch_counter += self.batch_size

        return batch_image, batch_metadata, batch_label

    def get_batch_eval(self):
        if self.batch_counter_val + self.batch_size < self.data_array.shape[0]:
            batch_image = self.data_array[
                self.batch_counter_val : self.batch_counter_val + self.batch_size, ...
            ]
            batch_label = self.data_label[
                self.batch_counter_val : self.batch_counter_val + self.batch_size, ...
            ]
            batch_metadata = self.meta_data[
                self.batch_counter_val : self.batch_counter_val + self.batch_size, ...
            ]
            self.batch_counter_val += self.batch_size
            # print(get_batch.BATCH_COUNTER)
        else:
            left_samples = self.data_array.shape[0] - self.batch_counter_val
            batch_image = self.data_array[
                self.batch_counter_val : self.batch_counter_val + left_samples, ...
            ]
            batch_label = self.data_label[
                self.batch_counter_val : self.batch_counter_val + left_samples, ...
            ]
            batch_metadata = self.meta_data[
                self.batch_counter_val : self.batch_counter_val + left_samples, ...
            ]
            self.batch_counter_val = 0

        return batch_image, batch_metadata, batch_label

    def shuffle_data(self):
        idx = np.arange(self.data_array.shape[0])
        np.random.shuffle(idx)
        self.data_array = self.data_array[idx, ...]
        self.data_label = self.data_label[idx, ...]
        self.meta_data = self.meta_data[idx, ...]

    # TODO aboid data replication
    def balance_data_by_replication(self):
        # labels, label_indexes = np.unique(self.data_label, return_inverse=True, axis=0)
        print(
            "Replicationg data from %s"
            % str(np.unique(self.data_label, return_counts=True))
        )
        labels = np.unique(self.data_label)
        max_label_count, _ = self.get_max_min_label_count()
        balanced_labels, balanced_images, balanced_features = [], [], []
        for i, l in enumerate(labels):
            # class_index = labels[label_indexes] == i
            class_index = np.where(self.data_label == l)[0]
            n_samples_this_class = class_index.shape[0]  # np.sum(class_index)
            if n_samples_this_class != max_label_count:
                balanced_labels.append(
                    self._oversampling_array(self.data_label[class_index])
                )
                balanced_images.append(
                    self._oversampling_array(self.data_array[class_index])
                )
                balanced_features.append(
                    self._oversampling_array(self.meta_data[class_index])
                )
            else:
                balanced_labels.append(self.data_label[class_index])
                balanced_images.append(self.data_array[class_index])
                balanced_features.append(self.meta_data[class_index])
        self.data_label = np.concatenate(balanced_labels)
        self.data_array = np.concatenate(balanced_images)
        self.meta_data = np.concatenate(balanced_features)
        print("to %s" % str(np.unique(self.data_label, return_counts=True)))

    def balance_data_by_replication_2_classes(self):
        self.balance_data_by_replication()

    def get_max_min_label_count(self):
        _, labels_count = np.unique(self.data_label, return_counts=True)
        return np.amax(labels_count), np.amin(labels_count)
