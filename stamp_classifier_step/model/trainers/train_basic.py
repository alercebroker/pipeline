import numpy as np
import os
import sys

PROJECT_PATH = os.path.join("..")
sys.path.append(PROJECT_PATH)
from models.classifiers.base_model import BaseModel as Model
from parameters import param_keys, general_keys

if __name__ == "__main__":
    seed_array = np.arange(10).tolist()

    data_path = os.path.join(
        PROJECT_PATH, "..", "datasets", "ZTF", "stamp_classifier", "ztf_dataset.pkl"
    )
    params = {param_keys.DATA_PATH_TRAIN: data_path, param_keys.CHANNELS_TO_USE: 1}

    accuracies = []
    for i in range(len(seed_array)):
        model = Model(params)
        metrics = model.fit()
        model.close()
        accuracies.append(metrics[general_keys.ACCURACY])
    print("Accuracies:\n %s" % str(accuracies))
    print(
        "\n %i models Test Accuracy: %.4f +/- %.4f"
        % (len(seed_array), float(np.mean(accuracies)), float(np.std(accuracies)))
    )
