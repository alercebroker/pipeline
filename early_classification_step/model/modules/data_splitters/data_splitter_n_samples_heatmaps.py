#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 00:17:54 2018
Data splitter by n samples to get
@author: asceta
"""
import numpy as np
from modules.data_splitters.data_splitter_n_samples import DatasetDividerInt


# Todo: return dicts instead of separate datasets
class DatasetDividerIntHeatmaps(DatasetDividerInt):
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

    def _train_test_split(self, data_array, data_label, test_size, random_state):
        samples_idxs = np.arange(data_array.shape[0])
        np.random.seed(random_state)
        np.random.shuffle(samples_idxs)
        test_idxs = samples_idxs[:test_size]
        train_idxs = samples_idxs[test_size:]
        result_dict = {
            "train_array": data_array[train_idxs],
            "test_array": data_array[test_idxs],
            "train_labels": data_label[train_idxs],
            "test_labels": data_label[test_idxs],
        }

        return (
            result_dict["train_array"],
            result_dict["test_array"],
            result_dict["train_labels"],
            result_dict["test_labels"],
        )
