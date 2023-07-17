import os
import sys
import numpy as np
import random

np.random.seed(0)
random.seed(0)

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
from models.classifiers.deepHits_nans_norm_crop_stamp_model import (
    DeepHiTSNanNormCropStampModel,
)
from models.classifiers.deepHits_entopy_reg_model import DeepHiTSEntropyRegModel
from trainers.base_trainer import Trainer
from parameters import param_keys, general_keys
import argparse

# parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_models_to_run",
    type=int,
    help="maximun number of models to run on this execution",
    required=False,
    default=2025,
)
parser.add_argument(
    "--position_in_comb_array",
    type=int,
    help="starting index in hyperparameter array to start model training",
    required=False,
    default=0,
)

if __name__ == "__main__":
    opts = parser.parse_args()
    entropy_reg_betas = [0, 0.3, 0.5, 0.8, 1.0]
    learning_rates = [1e-4, 5e-3, 1e-3, 5e-4, 5e-5]
    drop_rates = [0.5, 0.2, 0.8]
    batch_sizes = [32, 16, 64]
    crop_sizes = [21, 41, 63]
    filter_sizes = [3, 5, 7]

    comb = []

    for filter_size in filter_sizes:
        for input_size in crop_sizes:
            for drop_rate in drop_rates:
                for batch_size in batch_sizes:
                    for lr in learning_rates:
                        for beta in entropy_reg_betas:
                            comb.append(
                                {
                                    "beta": beta,
                                    "lr": lr,
                                    "drop_rate": drop_rate,
                                    "batch_size": batch_size,
                                    "input_size": input_size,
                                    "filter_size": filter_size,
                                }
                            )

    random.shuffle(comb)
    n_models_to_run = opts.n_models_to_run  # int(len(comb) % 1000)
    position_in_comb_array = (
        opts.position_in_comb_array
    )  # int((len(comb) / n_models_to_run) - 1)
    short_comb = comb[
        position_in_comb_array
        * n_models_to_run : (position_in_comb_array + 1)
        * n_models_to_run
    ]
    # print(len(short_comb))
    # print(position_in_comb_array * n_models_to_run)
    # print((position_in_comb_array + 1) * n_models_to_run)
    for aux_dict in short_comb[::-1]:
        beta = aux_dict["beta"]
        lr = aux_dict["lr"]
        drop_rate = aux_dict["drop_rate"]
        batch_size = aux_dict["batch_size"]
        input_size = aux_dict["input_size"]
        filter_size = aux_dict["filter_size"]
        if lr != 5e-5:
            continue

        command = (
            "python default_training.py --entropy-beta %.4f --learning-rate %.5f --dropout-rate %.4f "
            "--batch-size %d --input-size %d --filters-size %d"
            % (beta, lr, drop_rate, batch_size, input_size, filter_size)
        )
        os.system(command)
