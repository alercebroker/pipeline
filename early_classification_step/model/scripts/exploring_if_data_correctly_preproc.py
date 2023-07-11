import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_nans_norm_crop_stamp_model import (
    DeepHiTSNanNormCropStampModel,
)
from models.classifiers.deepHits_change_preprocessing_pipeline import (
    DeepHiTSCustomePreprocessing,
)
from models.classifiers.deepHits_entopy_reg_model import DeepHiTSEntropyRegModel
from trainers.base_trainer import Trainer
from parameters import param_keys, general_keys
from modules.data_loaders.ztf_preprocessor import ZTFDataPreprocessor
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    # data_path = os.path.join("../../pickles", 'training_set_Nov-26-2019.pkl')
    data_path = "../../pickles/converted_data.pkl"

    n_classes = 5
    params = {
        param_keys.RESULTS_FOLDER_NAME: "alerce_dataset_preprocessing",
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TRAIN_ITERATIONS_HORIZON: 30000,
        param_keys.TRAIN_HORIZON_INCREMENT: 10000,
        param_keys.TEST_SIZE: n_classes * 50,
        param_keys.VAL_SIZE: n_classes * 50,
        param_keys.NANS_TO: 0,
        param_keys.NUMBER_OF_CLASSES: n_classes,
        param_keys.CROP_SIZE: 21,
        param_keys.INPUT_IMAGE_SIZE: 21,
        param_keys.VALIDATION_MONITOR: general_keys.LOSS,
        param_keys.VALIDATION_MODE: general_keys.MIN,
        param_keys.ENTROPY_REG_BETA: 0.0,
    }
    mdl = DeepHiTSCustomePreprocessing(params)

    dataset = mdl._prepare_input(
        X=np.empty([]), y=np.empty([]), validation_data=None, test_data=None
    )
    train_x = dataset[0].data_array
    for idx in range(len(train_x)):
        print("\n", train_x[idx, ..., 0].min())
        print(train_x[idx, ..., 1].min())
        print(train_x[idx, ..., 2].min())
    for idx in range(len(train_x)):
        print("\n", train_x[idx, ..., 0].max())
        print(train_x[idx, ..., 1].max())
        print(train_x[idx, ..., 2].max())
    del mdl
    tf.reset_default_graph()

    data_preprocessor = ZTFDataPreprocessor(params)
    data_preprocessor.set_pipeline(
        [
            data_preprocessor.image_check_single_image,
            data_preprocessor.image_clean_misshaped,
            data_preprocessor.image_select_channels,
            data_preprocessor.image_normalize_by_image,
            data_preprocessor.image_nan_to_num,
            data_preprocessor.image_crop_at_center,
        ]
    )
    params.update({param_keys.INPUT_DATA_PREPROCESSOR: data_preprocessor})
    mdl = DeepHiTSCustomePreprocessing(params)

    dataset = mdl._prepare_input(
        X=np.empty([]), y=np.empty([]), validation_data=None, test_data=None
    )
    train_x = dataset[0].data_array
    for idx in range(len(train_x)):
        print("\n", train_x[idx, ..., 0].min())
        print(train_x[idx, ..., 1].min())
        print(train_x[idx, ..., 2].min())
    for idx in range(len(train_x)):
        print("\n", train_x[idx, ..., 0].max())
        print(train_x[idx, ..., 1].max())
        print(train_x[idx, ..., 2].max())
