import numpy as np
import pandas as pd
import os
import sys

PROJECT_PATH = os.path.join("..")
sys.path.append(PROJECT_PATH)
from models.visualizers.base_ae_visualizer import BaseAEVisualizer
from parameters import param_keys, general_keys


def filter_datadict_by_label(data_dict, chosen_label):
    # indexes = np.arange(data_dict[general_keys.IMAGES].shape[0])
    class_indexes = np.where(data_dict[general_keys.LABELS] == chosen_label)[0]
    new_data_dict = {
        general_keys.IMAGES: data_dict[general_keys.IMAGES][class_indexes],
        general_keys.LABELS: None,
    }


if __name__ == "__main__":
    seed_array = np.arange(1).tolist()

    data_path = os.path.join(
        PROJECT_PATH, "..", "pickles", "preprocessed_data_with_heatmaps21x21.pkl"
    )
    data_dict = pd.read_pickle(data_path)

    params = {
        param_keys.VAL_SIZE: 200,
    }

    checkpoint_path = os.path.join(
        PROJECT_PATH, "results/best_model_so_far/checkpoints", "model"
    )

    recons_error = []
    for i in range(len(seed_array)):
        model = BaseAEVisualizer(params)
        model.load_encoder(checkpoint_path)
        metrics = model.fit(
            X=data_dict[general_keys.IMAGES], y=data_dict[general_keys.HEATMAPS]
        )
        model.close()
        recons_error.append(metrics[general_keys.LOSS])
    print("Reconstruction Error:\n %s" % str(recons_error))
    print(
        "\n %i models Test Reconstruction Error: %.4f +/- %.4f"
        % (len(seed_array), float(np.mean(recons_error)), float(np.std(recons_error)))
    )
