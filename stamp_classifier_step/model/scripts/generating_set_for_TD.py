# %%
import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_with_features_entropy_reg import (
    DeepHiTSWithFeaturesEntropyReg,
)
from parameters import param_keys, general_keys
import numpy as np
import pickle
import pandas as pd

data_path = os.path.join(PROJECT_PATH, "../pickles", "converted_data.pkl")
# 'training_set_Nov-26-2019.pkl')
n_classes = 5
params = {
    param_keys.RESULTS_FOLDER_NAME: "aux_model",
    param_keys.DATA_PATH_TRAIN: data_path,
    param_keys.WAIT_FIRST_EPOCH: False,
    param_keys.N_INPUT_CHANNELS: 3,
    param_keys.CHANNELS_TO_USE: [0, 1, 2],
    param_keys.TRAIN_ITERATIONS_HORIZON: 30000,
    param_keys.TRAIN_HORIZON_INCREMENT: 10000,
    param_keys.TEST_SIZE: n_classes * 100,
    param_keys.VAL_SIZE: n_classes * 100,
    param_keys.NANS_TO: 0,
    param_keys.NUMBER_OF_CLASSES: n_classes,
    param_keys.CROP_SIZE: 63,
    param_keys.INPUT_IMAGE_SIZE: 63,
    param_keys.LEARNING_RATE: 0.0001,
    param_keys.VALIDATION_MONITOR: general_keys.LOSS,
    param_keys.VALIDATION_MODE: general_keys.MIN,
    param_keys.ENTROPY_REG_BETA: 0.5,
    param_keys.DROP_RATE: 0.5,
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
model = DeepHiTSWithFeaturesEntropyReg(params)

train_set, val_set, test_set = model._prepare_input(
    X=np.empty([]), y=np.empty([]), validation_data=None, test_data=None
)

print("Train %s" % str(np.unique(train_set.data_label, return_counts=True)))
print("Val %s" % str(np.unique(val_set.data_label, return_counts=True)))
print("Test %s" % str(np.unique(test_set.data_label, return_counts=True)))

dataset_dict = {
    "Train": {
        "images": train_set.data_array,
        "labels": train_set.data_label,
        "features": train_set.meta_data,
    },
    "Validation": {
        "images": val_set.data_array,
        "labels": val_set.data_label,
        "features": val_set.meta_data,
    },
    "Test": {
        "images": test_set.data_array,
        "labels": test_set.data_label,
        "features": test_set.meta_data,
    },
}

save_path = os.path.join(PROJECT_PATH, "..", "pickles", "td_ztf_stamp_17_06_20.pkl")
pickle.dump(dataset_dict, open(save_path, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
