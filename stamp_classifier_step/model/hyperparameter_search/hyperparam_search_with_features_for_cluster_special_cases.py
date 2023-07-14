import os
import sys
import numpy as np
import random

np.random.seed(0)
random.seed(0)

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)
import argparse
from modules.utils import get_folder_names_in_path

# parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--n_models_to_run",
    type=int,
    help="maximun number of models to run on this execution",
    required=False,
    default=100,
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
    entropy_reg_betas = [0.5]
    learning_rates = [1e-4, 5e-3, 1e-3, 5e-4]
    drop_rates = [0.5]
    batch_sizes = [32]
    crop_sizes = [21]
    filter_sizes = [3]

    comb = []
    comb.append(
        {
            "beta": 1.0,
            "lr": 5e-3,
            "drop_rate": 0.5,
            "batch_size": 64,
            "input_size": 63,
            "filter_size": 5,
        }
    )
    comb.append(
        {
            "beta": 0.5,
            "lr": 1e-3,
            "drop_rate": 0.8,
            "batch_size": 32,
            "input_size": 21,
            "filter_size": 7,
        }
    )
    comb.append(
        {
            "beta": 0.5,
            "lr": 5e-3,
            "drop_rate": 0.2,
            "batch_size": 64,
            "input_size": 63,
            "filter_size": 5,
        }
    )
    comb.append(
        {
            "beta": 0.8,
            "lr": 1e-3,
            "drop_rate": 0.8,
            "batch_size": 32,
            "input_size": 21,
            "filter_size": 3,
        }
    )
    comb.append(
        {
            "beta": 0.5,
            "lr": 5e-3,
            "drop_rate": 0.5,
            "batch_size": 64,
            "input_size": 21,
            "filter_size": 3,
        }
    )

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

    print(len(comb))
    # print(comb)
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
    print(len(short_comb))
    # print(short_comb)
    # command = "python table_generation/evaluating_models_to_get_results_pickle_with_features.py"
    # os.system(command)
    for aux_dict in short_comb[::-1]:
        beta = aux_dict["beta"]
        lr = aux_dict["lr"]
        drop_rate = aux_dict["drop_rate"]
        batch_size = aux_dict["batch_size"]
        input_size = aux_dict["input_size"]
        filter_size = aux_dict["filter_size"]

        command = (
            "python default_training_with_features.py --entropy-beta %.4f --learning-rate %.5f --dropout-rate %.4f "
            "--batch-size %d --input-size %d --filters-size %d"
            % (beta, lr, drop_rate, batch_size, input_size, filter_size)
        )
        os.system(command)
    # command = "cp -r ../results/hyperparameter_search_06_06_20_special_cases/* ../results/final_hyperparam_search_v1/"
    # os.system(command)
    # print('CP COMPLETE')
    # local_path = os.path.join(PROJECT_PATH, 'results',
    #                           'hyperparameter_search_06_06_20_special_cases')
    # target_path = os.path.join(PROJECT_PATH, 'results',
    #                            'final_hyperparam_search_v1')
    # local_path_folders = get_folder_names_in_path(local_path)
    # target_path_folders = get_folder_names_in_path(target_path)
    # for folder_name in local_path_folders:
    #   if folder_name not in target_path_folders:
    #     print(folder_name)
    # command = "python table_generation/evaluating_models_to_get_results_pickle_with_features_with_special_cases.py"
    # os.system(command)
