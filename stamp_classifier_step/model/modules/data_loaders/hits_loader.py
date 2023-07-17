from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import pickle as pkl

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)

from modules.data_set_generic import Dataset
from modules.data_splitters.data_splitter_percentages import DatasetDividerPercentage
from parameters import general_keys, param_keys

"""
hist2013 data loader
"""


# TODO: evaluate if it's good idea to pass params and use batchsize in
# dataset_generic
class HiTSLoader(object):
    """
    Constructor
    """

    def __init__(self, params: dict, label_value=1):
        self.path = params[param_keys.DATA_PATH_TRAIN]
        self.batch_size = params[param_keys.BATCH_SIZE]
        self.data_splitter = DatasetDividerPercentage(
            test_size=0.12, validation_size=0.08
        )
        self.first_n_samples = 125000
        self.label_value = label_value
        self.channel_to_get = 2

    def _init_splits_dict(self):
        datasets_dict = {
            general_keys.TRAIN: None,
            general_keys.VALIDATION: None,
            general_keys.TEST: None,
        }
        return datasets_dict

    def _load_data(self, path):
        infile = open(path, "rb")
        data = pkl.load(infile)
        return data

    def _get_first_n_samples_by_label(self, data_dict, n_samples, label_value):
        images = data_dict[general_keys.IMAGES]
        labels = data_dict[general_keys.LABELS]
        # print(labels.shape)
        label_value_idxs = np.where(labels == label_value)[0]
        # print(label_value_idxs.shape)
        np.random.shuffle(label_value_idxs)
        label_idxs_to_get = label_value_idxs[:n_samples]
        data_dict[general_keys.IMAGES] = images[label_idxs_to_get]
        data_dict[general_keys.LABELS] = labels[label_idxs_to_get]
        return data_dict

    def normalize_images(self, images):
        for image_index in range(images.shape[0]):
            image = images[image_index]
            image -= image.min()
            image = image / image.max()
            images[image_index] = image
        return images

    def get_datadict(self):
        data_dict = self._load_data(path)
        data_dict = self._get_first_n_samples_by_label(
            data_dict, n_samples=self.first_n_samples, label_value=self.label_value
        )
        # get difference image
        # Todo: code as param to get channel
        selected_image_channels = data_dict[general_keys.IMAGES][
            ..., self.channel_to_get
        ]
        if len(selected_image_channels.shape) == 3:
            selected_image_channels = selected_image_channels[..., np.newaxis]
        data_dict[general_keys.IMAGES] = selected_image_channels
        # normalice images
        data_dict[general_keys.IMAGES] = self.normalize_images(
            data_dict[general_keys.IMAGES]
        )
        return data_dict

    def get_preprocessed_datasets_splitted(self) -> dict:
        data_dict = self.get_datadict()
        dataset = Dataset(
            data_array=data_dict[general_keys.IMAGES],
            data_label=data_dict[general_keys.LABELS],
            batch_size=self.batch_size,
        )
        datasets_dict = self._init_splits_dict()
        self.data_splitter.set_dataset_obj(dataset)
        (
            train_dataset,
            test_dataset,
            val_dataset,
        ) = self.data_splitter.get_train_test_val_set_objs()
        datasets_dict[general_keys.TRAIN] = train_dataset
        datasets_dict[general_keys.TEST] = test_dataset
        datasets_dict[general_keys.VALIDATION] = val_dataset
        return datasets_dict


if __name__ == "__main__":
    params = {
        param_keys.DATA_PATH_TRAIN: os.path.join(
            PROJECT_PATH, "..", "datasets", "HiTS2013_300k_samples.pkl"
        ),
        param_keys.BATCH_SIZE: 50,
    }
    data_loader = HiTSLoader(params)
    datasets_dict = data_loader.get_preprocessed_datasets_splitted()
    print("train %s" % str(datasets_dict[general_keys.TRAIN].data_array.shape))
    print("test %s" % str(datasets_dict[general_keys.TEST].data_array.shape))
    print("val %s" % str(datasets_dict[general_keys.VALIDATION].data_array.shape))
