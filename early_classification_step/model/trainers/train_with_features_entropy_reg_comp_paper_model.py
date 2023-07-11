import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_with_features_entropy_reg import (
    DeepHiTSWithFeaturesEntropyReg,
)
from models.classifiers.deepHits_entopy_reg_model import DeepHiTSEntropyRegModel
from trainers.base_trainer import Trainer
from parameters import param_keys, general_keys
from modules.data_loaders.frame_to_input_with_features import FrameToInputWithFeatures

if __name__ == "__main__":

    N_TRAINS = 1
    DATA_PATH = os.path.join(PROJECT_PATH, "../pickles", "training_set_May-06-2020.pkl")
    # DATA_PATH = "../../pickles/converted_data.pkl"
    ENTROPY_REG_BETAS_LIST = [
        0.5,
    ]

    N_CLASSES = 5

    n_classes = 5
    params = {
        param_keys.RESULTS_FOLDER_NAME: "staging_model",
        param_keys.DATA_PATH_TRAIN: DATA_PATH,
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TRAIN_ITERATIONS_HORIZON: 30000,
        param_keys.TRAIN_HORIZON_INCREMENT: 10000,
        param_keys.TEST_SIZE: n_classes * 100,
        param_keys.VAL_SIZE: n_classes * 100,
        param_keys.NANS_TO: 0,
        param_keys.NUMBER_OF_CLASSES: n_classes,
        param_keys.CROP_SIZE: 21,
        param_keys.INPUT_IMAGE_SIZE: 21,
        param_keys.LEARNING_RATE: 0.001,
        param_keys.VALIDATION_MONITOR: general_keys.LOSS,
        param_keys.VALIDATION_MODE: general_keys.MIN,
        param_keys.ENTROPY_REG_BETA: 0.5,
        param_keys.DROP_RATE: 0.5,
        param_keys.BATCH_SIZE: 64,
        param_keys.KERNEL_SIZE: 5,
        param_keys.FEATURES_NAMES_LIST: [
            "sgscore1",
            "distpsnr1",
            "sgscore2",
            "distpsnr2",
            "sgscore3",
            "distpsnr3",
            "isdiffpos",
            "fwhm",
            "magpsf",
            "sigmapsf",
            "ra",
            "dec",
            "diffmaglim",
            "classtar",
            "ndethist",
            "ncovhist",
            "ecl_lat",
            "ecl_long",
            "gal_lat",
            "gal_long",
            "non_detections",
            "chinr",
            "sharpnr",
        ],
        param_keys.BATCHNORM_FEATURES_FC: True,
        param_keys.FEATURES_CLIPPING_DICT: {
            "sgscore1": [-1, "max"],
            "distpsnr1": [-1, "max"],
            "sgscore2": [-1, "max"],
            "distpsnr2": [-1, "max"],
            "sgscore3": [-1, "max"],
            "distpsnr3": [-1, "max"],
            "fwhm": ["min", 10],
            "distnr": [-1, "max"],
            "ndethist": ["min", 20],
            "ncovhist": ["min", 3000],
            "chinr": [-1, 15],
            "sharpnr": [-1, 1.5],
            "non_detections": ["min", 2000],
        },
    }
    aux_model = DeepHiTSWithFeaturesEntropyReg(params)
    aux_model._prepare_input()
    aux_model.close()
    del aux_model
    params.update({param_keys.DATA_PATH_TRAIN: "../../pickles/converted_data.pkl"})
    trainer = Trainer(params)

    for beta in ENTROPY_REG_BETAS_LIST:
        params.update({param_keys.ENTROPY_REG_BETA: beta})
        model_name = (
            "DeepHits_EntropyRegBeta%.4f_batch%d_lr%.5f_droprate%.4f_inputsize%d_filtersize%d"
            % (
                params[param_keys.ENTROPY_REG_BETA],
                params[param_keys.BATCH_SIZE],
                params[param_keys.LEARNING_RATE],
                params[param_keys.DROP_RATE],
                params[param_keys.INPUT_IMAGE_SIZE],
                params[param_keys.KERNEL_SIZE],
            )
        )
        # trainer.train_model_n_times(DeepHiTSWithFeaturesEntropyReg, params,
        trainer.train_model_n_times(
            DeepHiTSWithFeaturesEntropyReg,
            params,
            train_times=N_TRAINS,
            model_name=model_name,
        )
