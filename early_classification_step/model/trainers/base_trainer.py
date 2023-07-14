import numpy as np
import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.base_model import BaseModel as Model
from parameters import param_keys, general_keys
from modules.print_manager import PrintManager


class Trainer(object):
    """
    Constructor
    """

    def __init__(self, params={param_keys.RESULTS_FOLDER_NAME: ""}):
        self.all_models_acc = {}
        self.print_manager = PrintManager()
        self.model_path = os.path.join(
            PROJECT_PATH, "results", params[param_keys.RESULTS_FOLDER_NAME]
        )

    def train_model_n_times(self, ModelClass, params, train_times, model_name=None):
        seed_array = np.arange(train_times).tolist()
        accuracies = []
        for i in range(len(seed_array)):
            if model_name is None:
                aux_model = ModelClass(params)
                model_name = aux_model.model_name
                aux_model.close()
            model = ModelClass(params, model_name + "_%i" % i)
            metrics = model.fit()
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

    def print_all_accuracies(self):
        msg = ""
        for model_name in self.all_models_acc.keys():
            model_metrics = self.all_models_acc[model_name]
            msg += "\n %s Test Accuracy: %.4f +/- %.4f" % (
                model_name,
                model_metrics[general_keys.MEAN],
                model_metrics[general_keys.STD],
            )
        model_names = list(self.all_models_acc.keys())
        self.print_to_log(msg, "_".join(model_names))

    def print_to_log(self, msg, log_name):
        log_file = log_name + ".log"
        print = self.print_manager.verbose_printing(True)
        file = open(os.path.join(self.model_path, log_file), "a")
        self.print_manager.file_printing(file)
        print(msg)
        self.print_manager.close()
        file.close()


if __name__ == "__main__":
    data_path = os.path.join(
        PROJECT_PATH, "..", "datasets", "ZTF", "stamp_classifier", "ztf_dataset.pkl"
    )
    params = {
        param_keys.DATA_PATH_TRAIN: data_path,
    }
    trainer = Trainer()
    params.update({param_keys.CHANNELS_TO_USE: 0, param_keys.WAIT_FIRST_EPOCH: False})
    trainer.train_model_n_times(
        Model, params, train_times=10, model_name="BaseNoEpochW8Ch0"
    )
    params.update({param_keys.CHANNELS_TO_USE: 0, param_keys.WAIT_FIRST_EPOCH: True})
    trainer.train_model_n_times(
        Model, params, train_times=10, model_name="BaseEpochW8Ch0"
    )
    params.update({param_keys.CHANNELS_TO_USE: 1, param_keys.WAIT_FIRST_EPOCH: False})
    trainer.train_model_n_times(
        Model, params, train_times=10, model_name="BaseNoEpochW8Ch1"
    )
    params.update({param_keys.CHANNELS_TO_USE: 1, param_keys.WAIT_FIRST_EPOCH: True})
    trainer.train_model_n_times(
        Model, params, train_times=10, model_name="BaseEpochW8Ch1"
    )

    trainer.print_all_accuracies()
