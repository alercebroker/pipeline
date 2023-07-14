import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_4class_model import DeepHiTS4ClassModel
from models.classifiers.deepHits_4class_nans_model import DeepHiTS4ClassModelNan
from models.classifiers.deepHits_4class_norm_stamp_model import (
    DeepHiTS4ClassModelNormStamp,
)
from models.classifiers.deepHits_4class_nans_norm_stamp_model import (
    DeepHiTS4ClassModelNanNormStamp,
)
from trainers.base_trainer import Trainer
from parameters import param_keys

if __name__ == "__main__":
    # data_path = os.path.join("../../pickles", 'corrected_oids_alerts.pkl')
    #                          "all_alerts_training.pkl")
    data_path = "../../pickles/converted_data.pkl"
    params = {
        param_keys.RESULTS_FOLDER_NAME: "dh_nans_01",
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TRAIN_ITERATIONS_HORIZON: 10000,
        param_keys.TRAIN_HORIZON_INCREMENT: 10000,
        param_keys.TEST_SIZE: 200,
        param_keys.VAL_SIZE: 200,
        param_keys.NANS_TO: 0,
    }
    trainer = Trainer(params)

    trainer.train_model_n_times(
        DeepHiTS4ClassModelNan, params, train_times=10, model_name="DH4ClassNan0"
    )

    trainer.train_model_n_times(
        DeepHiTS4ClassModelNanNormStamp,
        params,
        train_times=10,
        model_name="DH4ClassNan0NormStamp",
    )

    params.update({param_keys.NANS_TO: 1})

    trainer.train_model_n_times(
        DeepHiTS4ClassModelNan, params, train_times=10, model_name="DH4ClassNan1"
    )

    trainer.train_model_n_times(
        DeepHiTS4ClassModelNanNormStamp,
        params,
        train_times=10,
        model_name="DH4ClassNan1NormStamp",
    )

    trainer.print_all_accuracies()
