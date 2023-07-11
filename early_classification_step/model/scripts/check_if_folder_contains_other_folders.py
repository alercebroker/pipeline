import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_with_features_atlas import DeepHiTSAtlasWithFeatures
import numpy as np
from parameters import param_keys, general_keys
from modules.data_loaders.atlas_preprocessor import ATLASDataPreprocessor
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from modules.utils import get_folder_names_in_path

from modules.confusion_matrix import plot_confusion_matrix

class_names = np.array(["cr", "streak", "burn", "scar", "kast", "spike", "noise"])


if __name__ == "__main__":
    local_path = os.path.join(PROJECT_PATH, "results", "hyperparameter_search_06_06_20")
    target_path = os.path.join(PROJECT_PATH, "results", "final_hyperparam_search_v1")
    local_path_folders = get_folder_names_in_path(local_path)
    target_path_folders = get_folder_names_in_path(target_path)
    for folder_name in local_path_folders:
        if folder_name not in target_path_folders:
            print(folder_name)
