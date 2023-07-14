import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_with_features_entropy_reg import (
    DeepHiTSWithFeaturesEntropyReg,
)
import numpy as np
from parameters import param_keys, general_keys

if __name__ == "__main__":
    N_TRAINS = 1
    DATA_PATH = os.path.join(PROJECT_PATH, "../pickles", "training_set_May-06-2020.pkl")
    # DATA_PATH = "../../pickles/converted_data.pkl"
    ENTROPY_REG_BETAS_LIST = [0, 0.3, 0.5, 0.8, 1.0]
    N_CLASSES = 5

    n_classes = 5
    params = {
        param_keys.RESULTS_FOLDER_NAME: "best_model_paper_beta_comparison",
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
        param_keys.ENTROPY_REG_BETA: 0.8,
        param_keys.DROP_RATE: 0.8,
        param_keys.BATCH_SIZE: 32,
        param_keys.KERNEL_SIZE: 3,
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
            "rb",
            "distnr",
            "magnr",
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
            "magnr": [-1, "max"],
            "ndethist": ["min", 20],
            "ncovhist": ["min", 3000],
            "chinr": [-1, 15],
            "sharpnr": [-1, 1.5],
            "non_detections": ["min", 2000],
        },
    }
    aux_model = DeepHiTSWithFeaturesEntropyReg(params)
    train_set, val_set, test_set = aux_model._prepare_input()
    results_folder_path = os.path.join(
        PROJECT_PATH, "results", "final_hyperparam_search"
    )
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
    print(os.path.abspath(results_folder_path))
    folders_in_path = [f.name for f in os.scandir(results_folder_path) if f.is_dir()]
    print(folders_in_path)
    test_acc = []
    val_acc = []
    for model_name_i in folders_in_path:
        if model_name in model_name_i:
            weights_path = os.path.join(
                results_folder_path, model_name_i, "checkpoints", "model"
            )
            print(weights_path)
            aux_model.load_model(weights_path)
            test_pred = aux_model.predict(test_set.data_array, test_set.meta_data)
            val_pred = aux_model.predict(val_set.data_array, val_set.meta_data)
            test_acc.append(np.mean(test_pred == test_set.data_label))
            val_acc.append(np.mean(val_pred == val_set.data_label))
    print("test")
    print(np.mean(test_acc), test_acc)
    print("val")
    print(np.mean(val_acc), val_acc)
