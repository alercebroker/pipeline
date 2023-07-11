import os
import sys
import pandas as pd
import numpy as np

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_4class_nans_norm_stamp_model import (
    DeepHiTS4ClassModelNanNormStamp,
)
from models.classifiers.deepHits_nans_norm_crop_stamp_model import (
    DeepHiTSNanNormCropStampModel,
)
from trainers.base_trainer import Trainer
from parameters import param_keys

if __name__ == "__main__":
    data_path = os.path.join("../../pickles", "training_set_with_bogus.pkl")

    n_classes = 5
    params = {
        param_keys.RESULTS_FOLDER_NAME: "crop_at_center",
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TRAIN_ITERATIONS_HORIZON: 10000,
        param_keys.TRAIN_HORIZON_INCREMENT: 10000,
        param_keys.TEST_SIZE: n_classes * 50,
        param_keys.VAL_SIZE: n_classes * 50,
        param_keys.NANS_TO: 1,
        param_keys.NUMBER_OF_CLASSES: n_classes,
        param_keys.CROP_SIZE: 21,
        param_keys.INPUT_IMAGE_SIZE: 21,
    }

    model = DeepHiTSNanNormCropStampModel(params)

    model.dataset_preprocessor.set_pipeline(
        [
            model.dataset_preprocessor.image_check_single_image,
            model.dataset_preprocessor.image_clean_misshaped,
        ]
    )

    train_set, val_set, test_set = model._data_init()
    raw_pickle = pd.read_pickle(
        os.path.join(PROJECT_PATH, "..", "pickles", "converted_data.pkl")
    )

    # 31727 samples
    images = np.array(raw_pickle["images"])
    labels = np.array(raw_pickle["labels"])
