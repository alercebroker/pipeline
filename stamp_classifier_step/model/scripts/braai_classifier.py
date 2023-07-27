import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_model import DeepHitsModel
from parameters import param_keys, general_keys
from trainers.base_trainer import Trainer


class BraaiTrainer(Trainer):
    """
    Constructor
    """

    def __init__(self, params={param_keys.RESULTS_FOLDER_NAME: ""}):
        super().__init__(params)

    def train_model_n_times(
        self,
        ModelClass,
        params,
        train_times,
        metadata_path,
        stamps_path,
        model_name=None,
    ):
        seed_array = np.arange(train_times).tolist()
        # load data
        df = pd.read_csv(metadata_path)
        triplets = np.load(stamps_path, mmap_mode="r")
        # split data
        print("total labels: ", np.unique(df.label, return_counts=True))
        x_train, x_test, y_train, y_test = train_test_split(
            triplets, df.label, test_size=0.1, random_state=42
        )
        print(
            "train-test shapes: ",
            x_train.shape,
            x_test.shape,
            y_train.shape,
            y_test.shape,
        )
        accuracies = []
        for i in range(len(seed_array)):
            if model_name is None:
                aux_model = ModelClass(params)
                model_name = aux_model.model_name
                aux_model.close()
            model = ModelClass(params, model_name + "_%i" % i)
            metrics = model.fit(X=x_train, y=y_train, test_data=(x_test, y_test))
            model.close()
            accuracies.append(metrics[general_keys.ACCURACY])
        self.print_to_log(
            "\n %i %s models Test Accuracy: %.4f +/- %.4f"
            % (len(seed_array), model_name, np.mean(accuracies), np.std(accuracies)),
            model_name,
        )
        self.all_models_acc[model_name] = {
            general_keys.MEAN: np.mean(accuracies),
            general_keys.STD: np.std(accuracies),
        }
        return accuracies


if __name__ == "__main__":
    stamps_path = os.path.join(
        PROJECT_PATH,
        "..",
        "datasets",
        "ZTF",
        "stamp_classifier",
        "braai",
        "triplets.norm.npy",
    )
    metadata_path = os.path.join(
        PROJECT_PATH,
        "..",
        "datasets",
        "ZTF",
        "stamp_classifier",
        "braai",
        "candidates.csv",
    )

    params = {
        param_keys.RESULTS_FOLDER_NAME: "braai",
        param_keys.DATA_PATH_TRAIN: None,
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TRAIN_ITERATIONS_HORIZON: 10000,
        param_keys.TRAIN_HORIZON_INCREMENT: 10000,
        param_keys.TEST_SIZE: None,
        param_keys.VAL_SIZE: 700,  # 10% of bogus
    }
    trainer = BraaiTrainer(params)

    trainer.train_model_n_times(
        DeepHitsModel,
        params,
        train_times=10,
        metadata_path=metadata_path,
        stamps_path=stamps_path,
        model_name="DHonBraai",
    )

    trainer.print_all_accuracies()
