import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)

import numpy as np
import shutil
from parameters import param_keys
import re


def string_to_int_or_float(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def parse_acronym_to_param_dict(acronym):
    values_as_str = re.findall("[-+]?\d*\.\d+|\d+", acronym)
    values = [string_to_int_or_float(s) for s in values_as_str]
    param_dict = {
        param_keys.ENTROPY_REG_BETA: values[0],
        param_keys.BATCH_SIZE: values[1],
        param_keys.LEARNING_RATE: values[2],
        param_keys.DROP_RATE: values[3],
        param_keys.CROP_SIZE: values[4],
        param_keys.INPUT_IMAGE_SIZE: values[4],
        param_keys.KERNEL_SIZE: values[5],
    }
    return param_dict


def get_folder_names_in_path(path):
    folder_names = [
        name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))
    ]
    return folder_names


def count_unique_folders_names(folder_names_list, erasing_chars_at_end=18):
    new_folder_names_list = [name[:-erasing_chars_at_end] for name in folder_names_list]
    # print(new_folder_names_list)
    # print(np.unique(new_folder_names_list, return_counts=True)
    unique_folder_names, unique_folder_names_count = np.unique(
        new_folder_names_list, return_counts=True
    )
    return unique_folder_names, unique_folder_names_count


def removing_incomplete_trains(results_path):
    all_trained_model_folder_name = get_folder_names_in_path(results_path)
    for trained_model_folder_name in all_trained_model_folder_name:
        trained_model_folder_path = os.path.join(
            results_path, trained_model_folder_name
        )
        training_completness_proof = os.path.join(
            trained_model_folder_path, "Test_cm_norm.png"
        )
        if not os.path.exists(training_completness_proof):
            shutil.rmtree(trained_model_folder_path)
            # print(trained_model_folder_name)


def removing_trains_to_have_only_N_trains(results_path, n_train_wanted=5):
    all_trained_model_folder_name = get_folder_names_in_path(results_path)
    #
    unique_folder_names, unique_folder_names_count = count_unique_folders_names(
        all_trained_model_folder_name
    )
    folder_names_above_5 = []
    folder_names_under_5 = []
    for i in range(len(unique_folder_names)):
        if unique_folder_names_count[i] < n_train_wanted:
            folder_names_under_5.append(unique_folder_names[i])
        elif unique_folder_names_count[i] > n_train_wanted:
            folder_names_above_5.append(unique_folder_names[i])
    #
    complete_folder_names_under_5 = [
        s
        for s in all_trained_model_folder_name
        if any(xs in s for xs in folder_names_under_5)
    ]
    # removing under 5 training files
    for specific_complete_folder_name_under_5 in complete_folder_names_under_5:
        shutil.rmtree(os.path.join(results_path, specific_complete_folder_name_under_5))
        # print(specific_complete_folder_name_under_5)
    # removing above 5 training files
    for specific_reduced_folder_name_above_5 in folder_names_above_5:
        # print(specific_reduced_folder_name_above_5)
        specific_complete_folder_name_above_5 = [
            s
            for s in all_trained_model_folder_name
            if any(xs in s for xs in [specific_reduced_folder_name_above_5])
        ]
        specific_complete_folder_name_above_5 = list(
            np.sort(specific_complete_folder_name_above_5)
        )
        # print(specific_complete_folder_name_above_5)
        for folder_name in specific_complete_folder_name_above_5:
            if len(specific_complete_folder_name_above_5) == n_train_wanted:
                break
            specific_complete_folder_name_above_5.remove(folder_name)
            shutil.rmtree(os.path.join(results_path, folder_name))
            # print(folder_name)


def get_learning_rate_corrected_folder_name(results_folder_path, folder_name):
    params_in_name = parse_acronym_to_param_dict(folder_name)
    # print(params_in_name)
    train_log_path = os.path.join(results_folder_path, folder_name, "train.log")
    train_log = open(train_log_path, "r")
    lines = train_log.readlines()
    for line in lines:
        # print(line)
        if "{'" in line:
            params_str = line
            params = eval(params_str)
            break
    split_folder_name = folder_name.split("_")
    for split_i in range(len(split_folder_name)):
        if "lr" in split_folder_name[split_i] and len(split_folder_name[split_i]) == 8:
            split_folder_name[split_i] = "lr%.5f" % params[param_keys.LEARNING_RATE]
            break
    new_folder_name = "_".join(split_folder_name)
    if params[param_keys.LEARNING_RATE] != params_in_name[param_keys.LEARNING_RATE]:
        print(
            "Missmatch of learning rates, changing from %s to %s"
            % (folder_name, new_folder_name)
        )
    return new_folder_name


def change_learning_rate_names_in_results(results_folder_path):
    all_trained_model_folder_name = get_folder_names_in_path(results_folder_path)
    for folder_name in all_trained_model_folder_name:
        new_folder_name = get_learning_rate_corrected_folder_name(
            results_folder_path, folder_name
        )
        old_folder_path = os.path.join(results_folder_path, folder_name)
        new_folder_path = os.path.join(results_folder_path, new_folder_name)
        os.rename(old_folder_path, new_folder_path)


if __name__ == "__main__":
    results_folder_name = "final_hyperparam_search"
    results_folder_path = os.path.join(PROJECT_PATH, "results", results_folder_name)

    # print(os.path.abspath(results_folder_path))
    all_trained_model_folder_name = get_folder_names_in_path(results_folder_path)
    print(count_unique_folders_names(all_trained_model_folder_name))
    removing_incomplete_trains(results_folder_path)
    change_learning_rate_names_in_results(results_folder_path)
    removing_trains_to_have_only_N_trains(results_folder_path)
    all_trained_model_folder_name = get_folder_names_in_path(results_folder_path)
    print(count_unique_folders_names(all_trained_model_folder_name))
