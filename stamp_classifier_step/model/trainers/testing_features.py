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
import tensorflow as tf


def remove_feature(feature_list, clipping_values, feature_to_remove):
    new_list = []
    for f in feature_list:
        if f == feature_to_remove:
            continue
        else:
            new_list.append(f)

    new_clip = {}
    for key, val in clipping_values.items():
        if key == feature_to_remove:
            continue
        else:
            new_clip[key] = val
    return new_list, new_clip


if __name__ == "__main__":

    N_TRAINS = 3
    DATA_PATH = os.path.join(PROJECT_PATH, "../pickles", "training_set_Jun-22-2020.pkl")
    # DATA_PATH = "../../pickles/converted_data.pkl"
    N_CLASSES = 5

    n_classes = 5
    feature_name_list = [
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
        "elong",
    ]

    clipping_values = {
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
    }

    params = {
        param_keys.RESULTS_FOLDER_NAME: "removing_features_testing",
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
        param_keys.KERNEL_SIZE: 3,
        param_keys.FEATURES_NAMES_LIST: feature_name_list,
        param_keys.BATCHNORM_FEATURES_FC: True,
        param_keys.FEATURES_CLIPPING_DICT: clipping_values,
    }

    trainer = Trainer(params)

    # model_name = 'AllFeatures_EntropyRegBeta%.4f_batch%d_lr%.5f_droprate%.4f_inputsize%d_filtersize%d' % \
    #                (params[param_keys.ENTROPY_REG_BETA],
    #                 params[param_keys.BATCH_SIZE],
    #                 params[param_keys.LEARNING_RATE],
    #                 params[param_keys.DROP_RATE],
    #                 params[param_keys.INPUT_IMAGE_SIZE],
    #                 params[param_keys.KERNEL_SIZE]
    #                 )
    # trainer.train_model_n_times(DeepHiTSWithFeaturesEntropyReg, params,
    #                             train_times=N_TRAINS,
    #                             model_name=model_name)
    #
    # """ Features to remove in order """
    # features_to_remove = ["distpsnr1", "distnr", "magnr", "magpsf", "sigmapsf"]
    #
    # for f_to_remove in features_to_remove:
    #   feature_name_list, clipping_values = remove_feature(feature_name_list, clipping_values, f_to_remove)
    #   print(feature_name_list, clipping_values)
    #   params.update({param_keys.FEATURES_NAMES_LIST: feature_name_list})
    #   params.update({param_keys.FEATURES_CLIPPING_DICT: clipping_values})
    #   model_name = f_to_remove+'_removed_EntropyRegBeta%.4f_batch%d_lr%.5f_droprate%.4f_inputsize%d_filtersize%d' % \
    #                (params[param_keys.ENTROPY_REG_BETA],
    #                 params[param_keys.BATCH_SIZE],
    #                 params[param_keys.LEARNING_RATE],
    #                 params[param_keys.DROP_RATE],
    #                 params[param_keys.INPUT_IMAGE_SIZE],
    #                 params[param_keys.KERNEL_SIZE]
    #                 )

    #   trainer = Trainer(params)
    #
    #   #tf.reset_default_graph()
    #
    #   trainer.train_model_n_times(DeepHiTSWithFeaturesEntropyReg, params=params,
    #                               train_times=N_TRAINS,
    #                               model_name=model_name)
    #
    # model_name = 'NoFeatures_EntropyRegBeta%.4f_batch%d_lr%.5f_droprate%.4f_inputsize%d_filtersize%d' % \
    #                (params[param_keys.ENTROPY_REG_BETA],
    #                 params[param_keys.BATCH_SIZE],
    #                 params[param_keys.LEARNING_RATE],
    #                 params[param_keys.DROP_RATE],
    #                 params[param_keys.INPUT_IMAGE_SIZE],
    #                 params[param_keys.KERNEL_SIZE]
    #                 )
    # trainer.train_model_n_times(DeepHiTSEntropyRegModel, params,
    #                             train_times=N_TRAINS,
    #                             model_name=model_name)

    feature_name_list, clipping_values = remove_feature(
        feature_name_list, clipping_values, "distnr"
    )
    feature_name_list, clipping_values = remove_feature(
        feature_name_list, clipping_values, "magnr"
    )
    feature_name_list, clipping_values = remove_feature(
        feature_name_list, clipping_values, "rb"
    )
    print(feature_name_list, clipping_values)
    params.update({param_keys.FEATURES_NAMES_LIST: feature_name_list})
    params.update({param_keys.FEATURES_CLIPPING_DICT: clipping_values})
    model_name = (
        "distnr-magnr-rb_removed_EntropyRegBeta%.4f_batch%d_lr%.5f_droprate%.4f_inputsize%d_filtersize%d"
        % (
            params[param_keys.ENTROPY_REG_BETA],
            params[param_keys.BATCH_SIZE],
            params[param_keys.LEARNING_RATE],
            params[param_keys.DROP_RATE],
            params[param_keys.INPUT_IMAGE_SIZE],
            params[param_keys.KERNEL_SIZE],
        )
    )
    trainer = Trainer(params)
    trainer.train_model_n_times(
        DeepHiTSWithFeaturesEntropyReg,
        params=params,
        train_times=N_TRAINS,
        model_name=model_name,
    )
