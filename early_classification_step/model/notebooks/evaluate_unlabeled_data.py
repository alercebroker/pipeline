import os, sys

PROJECT_PATH = os.path.join("..")
sys.path.append(PROJECT_PATH)
from parameters import param_keys, general_keys
from models.classifiers.deepHits_with_features_entropy_reg import (
    DeepHiTSWithFeaturesEntropyReg,
)
from modules.data_loaders.frame_to_input_with_features import FrameToInputWithFeatures
from models.classifiers.deepHits_entopy_reg_model import DeepHiTSEntropyRegModel
from modules.data_set_generic import Dataset
import tensorflow as tf


def get_predictions_of_chunk(
    chunk_data_path, training_set_path, model_path, with_features=True, kernel_size=3
):
    DATA_PATH = training_set_path
    N_CLASSES = 5

    features_keys = [
        "oid",
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
    ]

    params = param_keys.default_params.copy()
    params.update(
        {
            param_keys.RESULTS_FOLDER_NAME: "clipped_feat_and_batchnorm",
            param_keys.DATA_PATH_TRAIN: DATA_PATH,
            param_keys.WAIT_FIRST_EPOCH: False,
            param_keys.N_INPUT_CHANNELS: 3,
            param_keys.CHANNELS_TO_USE: [0, 1, 2],
            param_keys.TRAIN_ITERATIONS_HORIZON: 30000,
            param_keys.TRAIN_HORIZON_INCREMENT: 10000,
            param_keys.TEST_SIZE: N_CLASSES * 100,
            param_keys.VAL_SIZE: N_CLASSES * 100,
            param_keys.NANS_TO: 0,
            param_keys.NUMBER_OF_CLASSES: N_CLASSES,
            param_keys.CROP_SIZE: 21,
            param_keys.INPUT_IMAGE_SIZE: 21,
            param_keys.VALIDATION_MONITOR: general_keys.LOSS,
            param_keys.VALIDATION_MODE: general_keys.MIN,
            param_keys.ENTROPY_REG_BETA: 0.3,
            param_keys.LEARNING_RATE: 1e-4,
            param_keys.DROP_RATE: 0.5,
            param_keys.BATCH_SIZE: 32,
            param_keys.KERNEL_SIZE: kernel_size,
            param_keys.FEATURES_NAMES_LIST: features_keys,
            param_keys.BATCHNORM_FEATURES_FC: True,
            param_keys.FEATURES_CLIPPING_DICT: {
                "sgscore1": [-1, "max"],
                "distpsnr1": [-1, "max"],
                "sgscore2": [-1, "max"],
                "distpsnr2": [-1, "max"],
                "sgscore3": [-1, "max"],
                "distpsnr3": [-1, "max"],
                "fwhm": ["min", 10],
                "ndethist": ["min", 20],
                "ncovhist": ["min", 3000],
                "chinr": [-1, 15],
                "sharpnr": [-1, 1.5],
                "non_detections": ["min", 2000],
            },
        }
    )

    frame_to_input = FrameToInputWithFeatures(params)
    frame_to_input.set_dumping_data_to_pickle(False)
    frame_to_input.dataset_preprocessor.set_pipeline(
        [
            frame_to_input.dataset_preprocessor.image_check_single_image,
            frame_to_input.dataset_preprocessor.image_clean_misshaped,
            frame_to_input.dataset_preprocessor.image_select_channels,
            frame_to_input.dataset_preprocessor.image_crop_at_center,
            frame_to_input.dataset_preprocessor.image_normalize_by_image,
            frame_to_input.dataset_preprocessor.image_nan_to_num,
            frame_to_input.dataset_preprocessor.features_clip,
            frame_to_input.dataset_preprocessor.features_normalize,
        ]
    )

    _ = frame_to_input.get_preprocessed_datasets_splitted()

    DATA_PATH = chunk_data_path
    N_CLASSES = 5

    features_keys = [
        "oid",
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
    ]

    params = param_keys.default_params.copy()
    params.update(
        {
            param_keys.RESULTS_FOLDER_NAME: "clipped_feat_and_batchnorm",
            param_keys.DATA_PATH_TRAIN: DATA_PATH,
            param_keys.WAIT_FIRST_EPOCH: False,
            param_keys.N_INPUT_CHANNELS: 3,
            param_keys.CHANNELS_TO_USE: [0, 1, 2],
            param_keys.TRAIN_ITERATIONS_HORIZON: 30000,
            param_keys.TRAIN_HORIZON_INCREMENT: 10000,
            param_keys.TEST_SIZE: N_CLASSES * 100,
            param_keys.VAL_SIZE: N_CLASSES * 100,
            param_keys.NANS_TO: 0,
            param_keys.NUMBER_OF_CLASSES: N_CLASSES,
            param_keys.CROP_SIZE: 21,
            param_keys.INPUT_IMAGE_SIZE: 21,
            param_keys.VALIDATION_MONITOR: general_keys.LOSS,
            param_keys.VALIDATION_MODE: general_keys.MIN,
            param_keys.ENTROPY_REG_BETA: 0.3,
            param_keys.LEARNING_RATE: 1e-4,
            param_keys.DROP_RATE: 0.5,
            param_keys.BATCH_SIZE: 32,
            param_keys.KERNEL_SIZE: kernel_size,
            param_keys.FEATURES_NAMES_LIST: features_keys,
            param_keys.BATCHNORM_FEATURES_FC: True,
            param_keys.FEATURES_CLIPPING_DICT: {
                "sgscore1": [-1, "max"],
                "distpsnr1": [-1, "max"],
                "sgscore2": [-1, "max"],
                "distpsnr2": [-1, "max"],
                "sgscore3": [-1, "max"],
                "distpsnr3": [-1, "max"],
                "fwhm": ["min", 10],
                "ndethist": ["min", 20],
                "ncovhist": ["min", 3000],
                "chinr": [-1, 15],
                "sharpnr": [-1, 1.5],
                "non_detections": ["min", 2000],
            },
        }
    )

    data_loader = FrameToInputWithFeatures(params)
    data_loader.set_dumping_data_to_pickle(False)

    unlabeled_dataset = data_loader.get_raw_dataset_unsplitted()

    NEW_DATA_IMAGES = unlabeled_dataset.data_array
    NEW_DATA_FEATURES = unlabeled_dataset.meta_data
    NEW_DATA_LABELS = unlabeled_dataset.data_label

    NEW_DATA_DATASET = Dataset(
        data_array=NEW_DATA_IMAGES,
        data_label=NEW_DATA_LABELS,
        meta_data=NEW_DATA_FEATURES,
        batch_size=None,
    )

    preprocessor = frame_to_input.get_dataset_preprocessor()
    preprocessed_dataset = preprocessor.preprocess_dataset(NEW_DATA_DATASET)

    images, features = (
        preprocessed_dataset.data_array,
        preprocessed_dataset.meta_data[:, 1:],
    )
    oid = preprocessed_dataset.meta_data[:, 0]

    params.update(
        {
            param_keys.FEATURES_NAMES_LIST: features_keys[1:],
            param_keys.DATA_PATH_TRAIN: None,
        }
    )

    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=config)
    with graph.as_default():
        if with_features:
            model = DeepHiTSWithFeaturesEntropyReg(params, session=sess)
        else:
            model = DeepHiTSEntropyRegModel(params, session=sess)
        checkpoint_path_best_so_far = os.path.join(model_path, "checkpoints", "model")
        model.load_model(checkpoint_path_best_so_far)

    if with_features:
        y_pred = model.predict(images, features=features)
        y_prob = model.predict_proba(images, features=features)
    else:
        y_pred = model.predict(images)
        y_prob = model.predict_proba(images)

    return oid, y_pred, y_prob
