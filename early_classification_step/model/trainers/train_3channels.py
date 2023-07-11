import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_model import DeepHitsModel
from models.classifiers.base_model import BaseModel
from trainers.base_trainer import Trainer
from parameters import param_keys


if __name__ == "__main__":
    data_path = os.path.join(
        PROJECT_PATH, "..", "datasets", "ZTF", "stamp_classifier", "ztf_dataset.pkl"
    )
    params = {
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.N_INPUT_CHANNELS: 3,
    }
    trainer = Trainer()

    params.update(
        {param_keys.CHANNELS_TO_USE: [0, 1, 2], param_keys.WAIT_FIRST_EPOCH: False}
    )
    trainer.train_model_n_times(
        DeepHitsModel, params, train_times=10, model_name="DHNoEpochW8ChAll"
    )

    params.update(
        {param_keys.CHANNELS_TO_USE: [0, 1, 2], param_keys.WAIT_FIRST_EPOCH: True}
    )
    trainer.train_model_n_times(
        DeepHitsModel, params, train_times=10, model_name="DHEpochW8ChAll"
    )

    params.update(
        {param_keys.CHANNELS_TO_USE: [0, 1, 2], param_keys.WAIT_FIRST_EPOCH: False}
    )
    trainer.train_model_n_times(
        BaseModel, params, train_times=10, model_name="BaseNoEpochW8ChAll"
    )

    params.update(
        {param_keys.CHANNELS_TO_USE: [0, 1, 2], param_keys.WAIT_FIRST_EPOCH: True}
    )
    trainer.train_model_n_times(
        BaseModel, params, train_times=10, model_name="BaseEpochW8ChAll"
    )

    trainer.print_all_accuracies()
