import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.base_model import BaseModel
from models.classifiers.base_model_old import BaseModelOld
from trainers.base_trainer import Trainer
from parameters import param_keys


if __name__ == "__main__":
    data_path = os.path.join(
        PROJECT_PATH, "..", "datasets", "ZTF", "stamp_classifier", "ztf_dataset.pkl"
    )
    params = {
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.TRAIN_ITERATIONS_HORIZON: 1000,
        param_keys.N_INPUT_CHANNELS: 1,
        param_keys.CHANNELS_TO_USE: 1,
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.RESULTS_FOLDER_NAME: os.path.join("tests", "train_basic"),
    }
    trainer = Trainer()

    trainer.train_model_n_times(
        BaseModelOld, params, train_times=10, model_name="TEST_BaseOldNoEpochW8Ch1"
    )

    trainer.train_model_n_times(
        BaseModel, params, train_times=10, model_name="TEST_BaseNoEpochW8Ch1"
    )

    trainer.print_all_accuracies()
