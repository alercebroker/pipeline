import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_nans_norm_crop_stamp_model import (
    DeepHiTSNanNormCropStampModel,
)
from models.classifiers.deepHits_with_features_entropy_reg import (
    DeepHiTSWithFeaturesEntropyReg,
)
from trainers.base_trainer import Trainer
from parameters import param_keys

if __name__ == "__main__":
    data_path = os.path.join("../../pickles", "training_set_isdiffpos.pkl")
    # data_path = "../../pickles/converted_data.pkl"

    n_classes = 5
    params = {
        param_keys.RESULTS_FOLDER_NAME: "with_isdiffpos",
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TRAIN_ITERATIONS_HORIZON: 10000,
        param_keys.TRAIN_HORIZON_INCREMENT: 10000,
        param_keys.TEST_SIZE: n_classes * 50,
        param_keys.VAL_SIZE: n_classes * 50,
        param_keys.NANS_TO: 0,
        param_keys.NUMBER_OF_CLASSES: n_classes,
        param_keys.CROP_SIZE: 21,
        param_keys.INPUT_IMAGE_SIZE: 21,
        param_keys.FEATURES_NAMES_LIST: [
            "sgscore1",
            "distpsnr1",
            "sgscore2",
            "distpsnr2",
            "sgscore3",
            "distpsnr3",
            "isdiffpos",
        ],
        param_keys.ENTROPY_REG_COEF: 1e-3,
    }
    trainer = Trainer(params)

    trainer.train_model_n_times(
        DeepHiTSWithFeaturesEntropyReg,
        params,
        train_times=10,
        model_name="DeepHiTSWithFeaturesEntropyReg",
    )

    trainer.train_model_n_times(
        DeepHiTSNanNormCropStampModel,
        params,
        train_times=10,
        model_name="DHNan0NormStampWBogusCrop",
    )

    trainer.print_all_accuracies()
