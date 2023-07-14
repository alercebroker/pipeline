from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)

from modules.data_loaders.ztf_stamps_loader import ZTFLoader
from modules.data_splitters.data_splitter_n_samples import DatasetDividerInt
from parameters import general_keys, param_keys
from modules.data_loaders.atlas_preprocessor import ATLASDataPreprocessor
from modules.data_set_generic import Dataset

"""
ztf stamps data loader
"""


# TODO: evaluate if it's good idea to pass params and use batchsize in
# dataset_generic
class ATLASLoader(ZTFLoader):
    """
    Constructor
    """

    def __init__(self, params: dict):
        super().__init__(params)

    def _create_dataset_preprocessor(self, params):
        dataset_preprocessor = ATLASDataPreprocessor(params)
        dataset_preprocessor.set_pipeline(
            [
                dataset_preprocessor.image_check_single_image,
                # dataset_preprocessor.image_clean_misshaped,
                dataset_preprocessor.image_select_channels,
                dataset_preprocessor.image_crop_at_center,
                dataset_preprocessor.image_normalize_by_image_1_1,
                dataset_preprocessor.image_nan_to_num,
            ]
        )
        return dataset_preprocessor

    def _dict_to_dataset(self, data_dict: dict):
        keys = list(data_dict.keys())
        # random_idxs = np.random.randint(0, len(np.array(data_dict[general_keys.LABELS])), 100)
        # print(np.array(data_dict[general_keys.LABELS])[random_idxs])
        # print(np.array(data_dict['labels_str'])[random_idxs])
        if general_keys.FEATURES not in keys:
            dataset = Dataset(
                data_array=data_dict[general_keys.IMAGES],
                data_label=data_dict[general_keys.LABELS],
                batch_size=self.batch_size,
            )
        else:
            dataset = Dataset(
                data_array=data_dict[general_keys.IMAGES],
                data_label=data_dict[general_keys.LABELS],
                meta_data=data_dict[general_keys.FEATURES],
                batch_size=self.batch_size,
            )
        return dataset


if __name__ == "__main__":
    n_classes = 7
    parameters = {
        param_keys.DATA_PATH_TRAIN: os.path.join(
            PROJECT_PATH, "..", "ATLAS", "atlas_data.pkl"
        ),
        param_keys.TEST_SIZE: 100 * n_classes,
        param_keys.VAL_SIZE: 50 * n_classes,
        param_keys.BATCH_SIZE: 50,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.NANS_TO: 1,
        param_keys.CROP_SIZE: None,
        param_keys.TEST_RANDOM_SEED: 42,
        param_keys.VALIDATION_RANDOM_SEED: 42,
    }
    data_loader = ATLASLoader(parameters)
    datasets_dict = data_loader.get_preprocessed_datasets_splitted()
    print("train %s" % str(datasets_dict[general_keys.TRAIN].data_array.shape))
    print("test %s" % str(datasets_dict[general_keys.TEST].data_array.shape))
    print("val %s" % str(datasets_dict[general_keys.VALIDATION].data_array.shape))

    # replication test
    train_dataset = datasets_dict[general_keys.TRAIN]
    print(
        "train samples %s, train labels %s"
        % (
            str(train_dataset.data_array.shape),
            np.unique(train_dataset.data_label, return_counts=True),
        )
    )
    train_dataset.balance_data_by_replication()
    print(
        "train samples %s, train labels %s"
        % (
            str(train_dataset.data_array.shape),
            np.unique(train_dataset.data_label, return_counts=True),
        )
    )
