import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_atlas import DeepHiTSAtlas
from modules.confusion_matrix import plot_confusion_matrix
from parameters import param_keys, general_keys
from modules.data_loaders.atlas_preprocessor import ATLASDataPreprocessor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from modules.data_set_generic import Dataset
from scripts.ATLAS.evaluate_and_plot_galactic_plane_model_only_imgs import (
    galactic_coordinates,
    plot_galactic_plane_of_blinds,
    plot_class_atlas_hist,
)
from modules.utils import save_pickle

class_names = np.array(["cr", "streak", "burn", "scar", "kast", "spike", "noise"])


if __name__ == "__main__":
    data_path = os.path.join(PROJECT_PATH, "..", "ATLAS", "atlas_data.pkl")
    n_classes = 7
    params = {
        param_keys.BATCH_SIZE: 32,
        param_keys.RESULTS_FOLDER_NAME: "testing_atlas",
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.N_INPUT_CHANNELS: 1,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TRAIN_ITERATIONS_HORIZON: 5000,
        param_keys.TRAIN_HORIZON_INCREMENT: 1000,
        param_keys.TEST_SIZE: n_classes * 50,
        param_keys.VAL_SIZE: n_classes * 50,
        param_keys.NANS_TO: 0,
        param_keys.NUMBER_OF_CLASSES: n_classes,
        param_keys.CROP_SIZE: 63,
        param_keys.INPUT_IMAGE_SIZE: 63,
        param_keys.VALIDATION_MONITOR: general_keys.LOSS,
        param_keys.VALIDATION_MODE: general_keys.MIN,
        param_keys.ENTROPY_REG_BETA: 0.5,
        param_keys.ITERATIONS_TO_VALIDATE: 10,
    }
    data_preprocessor = ATLASDataPreprocessor(params)
    pipeline = [
        data_preprocessor.image_check_single_image,
        data_preprocessor.image_clean_misshaped,
        data_preprocessor.image_select_channels,
        data_preprocessor.images_to_gray_scale,
        data_preprocessor.image_crop_at_center,
        data_preprocessor.image_normalize_by_image,
        data_preprocessor.image_nan_to_num,
    ]
    data_preprocessor.set_pipeline(pipeline)
    params.update({param_keys.INPUT_DATA_PREPROCESSOR: data_preprocessor})

    model = DeepHiTSAtlas(params)
    train_set, val_set, test_set = model.get_dataset_used_for_training()
    test_pred = model.predict(test_set.data_array)
    plot_confusion_matrix(test_set.data_label, test_pred, show=True, title="Random")

    checkpoint_path = os.path.join(
        PROJECT_PATH,
        "results",
        "testing_atlas",
        "DeepHitsAtlas_20200628-154500",
        "checkpoints",
        "model",
    )
    model.load_model(checkpoint_path)
    # model.fit()
    test_pred = model.predict(test_set.data_array)
    plot_confusion_matrix(test_set.data_label, test_pred, show=True)

    blind_data = pd.read_pickle(os.path.join(PROJECT_PATH, "../ATLAS/atlas_blind.pkl"))
    blind_images = blind_data[general_keys.IMAGES]
    blind_features = blind_data[general_keys.FEATURES]
    # print('')
    # save_pickle(blind_data[general_keys.FEATURES], 'blind_features.pkl')
    blind_dataset = Dataset(blind_images, None, None, meta_data=blind_features)
    print(blind_dataset.data_array.shape)
    blind_dataset_preprop = model.dataset_preprocessor.preprocess_dataset(blind_dataset)
    print(blind_dataset_preprop.data_array.shape)
    blind_preds = model.predict(blind_dataset_preprop.data_array)
    plot_class_atlas_hist(blind_preds, class_names_short=class_names)
    plot_galactic_plane_of_blinds(blind_dataset_preprop, blind_preds, class_names)

    # a = pd.read_pickle(
    #     '../../tests/small_test_with_custome_features_and_metadata.pkl')
