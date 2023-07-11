import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_atlas import DeepHiTSAtlas
from trainers.base_trainer import Trainer
from parameters import param_keys, general_keys
from modules.data_loaders.atlas_preprocessor import ATLASDataPreprocessor

if __name__ == "__main__":
    N_TRAIN = 1
    data_path = os.path.join(PROJECT_PATH, "..", "ATLAS", "atlas_data.pkl")
    n_classes = 7
    params = {
        param_keys.BATCH_SIZE: 32,
        param_keys.RESULTS_FOLDER_NAME: "atlas_basic_norm_comp",
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TRAIN_ITERATIONS_HORIZON: 5000,
        param_keys.TRAIN_HORIZON_INCREMENT: 1000,
        param_keys.TEST_SIZE: n_classes * 50,
        param_keys.VAL_SIZE: n_classes * 50,
        param_keys.NANS_TO: 0,
        param_keys.NUMBER_OF_CLASSES: n_classes,
        param_keys.CROP_SIZE: None,
        param_keys.INPUT_IMAGE_SIZE: 101,
        param_keys.VALIDATION_MONITOR: general_keys.LOSS,
        param_keys.VALIDATION_MODE: general_keys.MIN,
        param_keys.ENTROPY_REG_BETA: 0.5,
        param_keys.ITERATIONS_TO_VALIDATE: 10,
    }

    trainer = Trainer(params)
    # trainer.train_model_n_times(DeepHiTSAtlas, params,
    #                             train_times=N_TRAIN,
    #                             model_name='DeepHitsAtlasRaw')

    params.update(
        {
            param_keys.CROP_SIZE: 63,
            param_keys.INPUT_IMAGE_SIZE: 63,
        }
    )
    data_preprocessor = ATLASDataPreprocessor(params)
    pipeline = [
        data_preprocessor.image_check_single_image,
        data_preprocessor.image_select_channels,
        data_preprocessor.images_to_gray_scale,
        data_preprocessor.image_crop_at_center,
        data_preprocessor.image_normalize_by_image,
        data_preprocessor.image_nan_to_num,
    ]
    data_preprocessor.set_pipeline(pipeline)
    params.update(
        {
            param_keys.INPUT_DATA_PREPROCESSOR: data_preprocessor,
            param_keys.N_INPUT_CHANNELS: 1,
        }
    )

    trainer.train_model_n_times(
        DeepHiTSAtlas, params, train_times=N_TRAIN, model_name="DeepHitsAtlasGray_Crop"
    )
