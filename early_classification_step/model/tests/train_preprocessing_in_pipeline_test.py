import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.base_preprocessing_in_pipeline_model import (
    BaseModelPreprocessingInPipeline,
)
from models.classifiers.base_model_old import BaseModelOld
from trainers.base_trainer import Trainer
from parameters import param_keys

if __name__ == "__main__":
    data_path = os.path.join(
        PROJECT_PATH, "..", "datasets", "ZTF", "stamp_classifier", "ztf_dataset.pkl"
    )
    params = {
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.RESULTS_FOLDER_NAME: os.path.join(
            "tests", "train_preprocessinf_in_pipeline"
        ),
    }
    trainer = Trainer()

    trainer.train_model_n_times(
        BaseModelPreprocessingInPipeline,
        params,
        train_times=20,
        model_name="TEST_BaseInPipeNoEpochW8Ch1",
    )

    trainer.train_model_n_times(
        BaseModelPreprocessingInPipeline,
        params,
        train_times=20,
        model_name="TEST_BaseInPipeNoEpochW8Ch1v2",
    )

    trainer.train_model_n_times(
        BaseModelOld, params, train_times=20, model_name="TEST_BaseOldNoEpochW8Ch1"
    )

    trainer.train_model_n_times(
        BaseModelOld, params, train_times=20, model_name="TEST_BaseOldNoEpochW8Ch1v2"
    )

    trainer.print_all_accuracies()
