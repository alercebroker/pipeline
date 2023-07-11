import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_atlas import DeepHiTSAtlas
from trainers.base_trainer import Trainer
from parameters import param_keys, general_keys
from modules.data_loaders.atlas_preprocessor import ATLASDataPreprocessor
import pandas as pd
from modules.data_set_generic import Dataset


if __name__ == "__main__":
    N_TRAIN = 1
    data_path = os.path.join(PROJECT_PATH, "..", "ATLAS", "atlas_data.pkl")
    n_classes = 7
    params = {
        param_keys.BATCH_SIZE: 32,
        param_keys.RESULTS_FOLDER_NAME: "atlas_basic_comp_norm_01",
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TRAIN_ITERATIONS_HORIZON: 1000,
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

    trainer = Trainer(params)
    params.update(
        {
            param_keys.TEST_SIZE: 3 * 50,
            param_keys.VAL_SIZE: 3 * 50,
            param_keys.NUMBER_OF_CLASSES: 3,
            param_keys.N_INPUT_CHANNELS: 3,
        }
    )
    params.update(
        {
            param_keys.CROP_SIZE: 63,
            param_keys.INPUT_IMAGE_SIZE: 63,
        }
    )
    data_preprocessor = ATLASDataPreprocessor(params)
    pipeline = [
        data_preprocessor.image_check_single_image,
        data_preprocessor.image_clean_misshaped,
        data_preprocessor.image_select_channels,
        data_preprocessor.images_to_gray_scale,
        data_preprocessor.image_crop_at_center,
        data_preprocessor.image_normalize_by_image,
        data_preprocessor.image_nan_to_num,
        data_preprocessor.labels_to_kast_streaks_artifact,
    ]
    data_preprocessor.set_pipeline(pipeline)
    params.update(
        {
            param_keys.INPUT_DATA_PREPROCESSOR: data_preprocessor,
            param_keys.N_INPUT_CHANNELS: 1,
        }
    )

    model = DeepHiTSAtlas(params, model_name="DeepHitsAtlas3Classes_Crop_Gray")
    model.fit()
    blind_data = pd.read_pickle(os.path.join(PROJECT_PATH, "../ATLAS/atlas_blind.pkl"))
    blind_images = blind_data[general_keys.IMAGES]
    blind_dataset = Dataset(blind_images, None, None)
    blind_dataset_preprop = model.dataset_preprocessor.preprocess_dataset(blind_dataset)

    blind_preds = model.predict(blind_dataset_preprop.data_array)

    import matplotlib.pyplot as plt
    import numpy as np

    class_names_short = np.array(["artifact", "kast", "streak"])
    label_values, label_counts = np.unique(blind_preds, return_counts=True)
    plt.bar(label_values, label_counts, align="center", label="Blind data distribution")
    plt.xticks(label_values, class_names_short[label_values], rotation=90)
    # plt.xticks(range(len_nonzero),
    #            indices[:len_nonzero], rotation=90)
    plt.xlim([-1, len(label_values)])
    plt.legend()
    plt.show()
