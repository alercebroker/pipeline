from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np

PROJECT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.append(PROJECT_PATH)

from modules.data_loaders.old.ztf_stamps_loader_old import ZTFLoaderOld
from modules.data_splitters.data_splitter_n_samples import DatasetDividerInt
from parameters import general_keys, param_keys
from modules.data_loaders.old.ztf_preprocessor_cascade import ZTFDataPreprocessorCascade

"""
ztf stamps data loader
"""


# TODO: evaluate if it's good idea to pass params and use batchsize in
# dataset_generic
class ZTFLoaderCascade(ZTFLoaderOld):
    """
    Constructor
    """

    def __init__(self, params: dict):
        self.path = params[param_keys.DATA_PATH_TRAIN]
        self.batch_size = params[param_keys.BATCH_SIZE]
        self.data_splitter = DatasetDividerInt(
            test_size=params[param_keys.TEST_SIZE],
            validation_size=params[param_keys.VAL_SIZE],
        )
        self.channel_to_get = params[param_keys.CHANNELS_TO_USE]

    def _get_preprocessed_dataset(self, data_dict):
        dataset = self._dict_to_dataset(data_dict)
        dataset_prepocessor = (
            ZTFDataPreprocessorCascade(dataset)
            .clean_misshaped()
            .image_select_channels(self.channel_to_get)
            .image_clean_nans()
            .image_normalize_by_sample()
        )
        dataset_prepocessed = dataset_prepocessor.get_dataset()
        return dataset_prepocessed


if __name__ == "__main__":
    parameters = {
        param_keys.DATA_PATH_TRAIN: os.path.join(
            PROJECT_PATH, "..", "datasets", "ZTF", "stamp_classifier", "ztf_dataset.pkl"
        ),
        param_keys.TEST_SIZE: 100,
        param_keys.VAL_SIZE: 50,
        param_keys.BATCH_SIZE: 50,
        param_keys.CHANNELS_TO_USE: 0,
    }
    data_loader = ZTFLoaderCascade(parameters)
    datasets_dict = data_loader.get_preprocessed_datasets_splitted()
    print("train %s" % str(datasets_dict[general_keys.TRAIN].data_array.shape))
    print("test %s" % str(datasets_dict[general_keys.TEST].data_array.shape))
    print("val %s" % str(datasets_dict[general_keys.VALIDATION].data_array.shape))

    # replication test
    train_dataset = datasets_dict[general_keys.TRAIN]
    print(
        "train samples %s, train SNe %i"
        % (str(train_dataset.data_array.shape), int(np.sum(train_dataset.data_label)))
    )
    train_dataset.balance_data_by_replication_2_classes()
    print(
        "train samples %s, train SNe %i"
        % (str(train_dataset.data_array.shape), int(np.sum(train_dataset.data_label)))
    )
