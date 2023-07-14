import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)

from parameters import param_keys, general_keys
import numpy as np
from modules.data_set_generic import Dataset
from modules.data_loaders.frame_to_input_with_features import FrameToInputWithFeatures

if __name__ == "__main__":
    DATA_PATH = os.path.join(PROJECT_PATH, "..", "pickles/training_set_Apr-27-2020.pkl")
    # 'pickles/converted_data.pkl')
    N_CLASSES = 5
    params = param_keys.default_params.copy()
    params.update(
        {
            param_keys.RESULTS_FOLDER_NAME: "gal_ecl_coord_beta_search",
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
            param_keys.ENTROPY_REG_BETA: 0.5,
            param_keys.LEARNING_RATE: 1e-4,
            param_keys.DROP_RATE: 0.5,
            param_keys.BATCH_SIZE: 32,
            param_keys.KERNEL_SIZE: 3,
            param_keys.FEATURES_NAMES_LIST: [
                "non_detections",
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
                "oid",
            ],
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
    )
    frame_to_input = FrameToInputWithFeatures(params)
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
    frame_to_input.set_dumping_data_to_pickle(dump_to_pickle=False)

    print("\n test preprocessed features")
    preprocessed_splits = frame_to_input.get_preprocessed_datasets_splitted()

    NEW_DATA_IMAGES = np.ones_like(preprocessed_splits[general_keys.TRAIN].data_array)
    NEW_DATA_FEATURES = np.ones_like(preprocessed_splits[general_keys.TRAIN].meta_data)

    NEW_DATA_DATASET = Dataset(
        data_array=NEW_DATA_IMAGES, data_label=None, meta_data=NEW_DATA_FEATURES
    )

    preprocessor = frame_to_input.get_dataset_preprocessor()
    preprocessed_dataset = preprocessor.preprocess_dataset(NEW_DATA_DATASET)
