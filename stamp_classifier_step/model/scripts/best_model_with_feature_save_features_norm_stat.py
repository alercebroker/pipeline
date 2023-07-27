import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_with_features_entropy_reg import (
    DeepHiTSWithFeaturesEntropyReg,
)
from trainers.base_trainer import Trainer
from parameters import param_keys, general_keys

import argparse


if __name__ == "__main__":
    argv = sys.argv[1:]
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--entropy-beta",
        type=float,
        default=0.5,
        help="regularization constant value beta",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-3, help="learning rate"
    )
    parser.add_argument("--dropout-rate", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--input-size", type=int, default=21, help="input image size")
    parser.add_argument(
        "--filters-size", type=int, default=3, help="size of convolutional filters"
    )

    args = parser.parse_args(argv)
    args = vars(args)

    # data_path = os.path.join("../../pickles", 'training_set_May-06-2020.pkl')
    data_path = "../../pickles/converted_data.pkl"

    n_classes = 5
    params = {
        param_keys.RESULTS_FOLDER_NAME: "training_with_feature_test",
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
        param_keys.CROP_SIZE: args["input_size"],
        param_keys.INPUT_IMAGE_SIZE: args["input_size"],
        param_keys.LEARNING_RATE: args["learning_rate"],
        param_keys.VALIDATION_MONITOR: general_keys.LOSS,
        param_keys.VALIDATION_MODE: general_keys.MIN,
        param_keys.ENTROPY_REG_BETA: args["entropy_beta"],
        param_keys.DROP_RATE: args["dropout_rate"],
        param_keys.BATCH_SIZE: args["batch_size"],
        param_keys.KERNEL_SIZE: args["filters_size"],
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

    model_name = (
        "DeepHits_EntropyRegBeta%.4f_batch%d_lr%.5f_droprate%.4f_inputsize%d_filtersize%d"
        % (
            args["entropy_beta"],
            args["batch_size"],
            args["learning_rate"],
            args["dropout_rate"],
            args["input_size"],
            args["filters_size"],
        )
    )

    print(model_name)

    model = DeepHiTSWithFeaturesEntropyReg(params, model_name)
    train, val, test = model._prepare_input()
    model.load_model(
        os.path.join(
            PROJECT_PATH, "results", "best_model_so_far", "checkpoints", "model"
        )
    )
    model.save_model(os.path.join(PROJECT_PATH, "results", "best_model_so_far"))
    model.evaluate(val.data_array, val.meta_data, val.data_label)
    model.evaluate(test.data_array, test.meta_data, test.data_label)
    _, raw_val, _ = model.data_loader.get_raw_datasets_splitted().values()
    raw_val = model.dataset_preprocessor.preprocess_dataset(raw_val)
    model.evaluate(raw_val.data_array, raw_val.meta_data, raw_val.data_label)
    _, raw_val, _ = model.data_loader.get_raw_datasets_splitted().values()
    model.close()
    del model
    aux_model = DeepHiTSWithFeaturesEntropyReg(params, model_name)
    aux_model.load_model_and_feature_stats(
        os.path.join(PROJECT_PATH, "results", "best_model_so_far")
    )
    val = aux_model.dataset_preprocessor.preprocess_dataset(raw_val)
    aux_model.evaluate(val.data_array, val.meta_data, val.data_label)
