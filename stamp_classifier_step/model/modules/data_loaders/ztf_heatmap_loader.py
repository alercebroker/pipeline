from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)

from modules.data_loaders.ztf_stamps_loader import ZTFLoader
from modules.data_set_generic import Dataset
from modules.data_splitters.data_splitter_n_samples import DatasetDividerInt
from parameters import general_keys, param_keys
from modules.data_loaders.ztf_preprocessor import ZTFDataPreprocessor

"""
ztf stamps data loader
"""


# TODO: evaluate if it's good idea to pass params and use batchsize in
# dataset_generic
class HeatmapLoader(ZTFLoader):
    """
    Constructor
    """

    def __init__(self, params: dict):
        super().__init__(params)

    def _dict_to_dataset(self, data_dict):
        dataset = Dataset(
            data_array=data_dict[general_keys.IMAGES],
            data_label=data_dict[general_keys.LABELS],
            meta_data=data_dict[general_keys.HEATMAPS],
            batch_size=self.batch_size,
        )
        return dataset
