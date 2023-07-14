import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_nans_norm_crop_stamp_model import (
    DeepHiTSNanNormCropStampModel,
)
from models.classifiers.deepHits_entopy_reg_model import DeepHiTSEntropyRegModel
from trainers.base_trainer import Trainer
from parameters import param_keys, general_keys

if __name__ == "__main__":
    data_path = os.path.join("../../pickles", "training_set_Apr-13-2020.pkl")
    # data_path = "../../pickles/converted_data.pkl"

    n_classes = 5
    params = {
        param_keys.RESULTS_FOLDER_NAME: "more_sn_dataset",
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
        param_keys.ENTROPY_REG_BETA: None,
        param_keys.BATCH_SIZE: 32,
        param_keys.LEARNING_RATE: 1e-4,
        param_keys.DROP_RATE: 0.5,
        param_keys.KERNEL_SIZE: 3,
    }
    trainer = Trainer(params)

    entropy_reg_betas = [
        0.5,
    ]  # [0.1, 0.3, 0.5, 0.8, 1.0]

    for entropy_reg_beta in entropy_reg_betas:
        params.update({param_keys.ENTROPY_REG_BETA: entropy_reg_beta})

        trainer.train_model_n_times(
            DeepHiTSEntropyRegModel,
            params,
            train_times=10,
            model_name="DeepHitsEntropyRegBeta%.4f"
            % params[param_keys.ENTROPY_REG_BETA],
        )

    params.update({param_keys.ENTROPY_REG_BETA: None})

    trainer.train_model_n_times(
        DeepHiTSNanNormCropStampModel,
        params,
        train_times=10,
        model_name="DHNan0NormStampWBogusCropValLoss",
    )

    trainer.print_all_accuracies()
