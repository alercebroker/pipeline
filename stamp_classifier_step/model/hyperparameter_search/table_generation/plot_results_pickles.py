import os
import sys

"""
first run ecaluating_models_to_get_results_pickle.py
and then this 
"""

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)

import matplotlib

# matplotlib.use('Agg')
import numpy as np
import pandas as pd
from parameters import general_keys
import matplotlib.pyplot as plt
from hyperparameter_search.table_generation.imposing_folder_and_trainings_consistency import (
    string_to_int_or_float,
)
import re
from scipy import stats

matplotlib.rcParams.update({"errorbar.capsize": 2})


def parse_acronym_to_param_dict_abbreviation(acronym):
    values_as_str = re.findall("[-+]?\d*\.\d+|\d+", acronym)
    values = [string_to_int_or_float(s) for s in values_as_str]
    param_dict = {
        "beta": values[0],
        "BS": values[1],
        "LR": "{:.0e}".format(values[2]),
        "DR": values[3],
        "IS": values[4],
        "KS": values[5],
    }
    return param_dict


def parse_param_dict_to_str(param_dict: dict) -> str:
    string = r"("
    for key in param_dict.keys():
        if key == "beta":
            string += r"$\beta$: " + str(param_dict[key]) + ", "
        else:
            string += str(key) + ": " + str(param_dict[key]) + ", "
        if key == "LR":
            string = string[:-1]
            string += "\n"
    string = string[:-2]
    string += ")"
    return string


if __name__ == "__main__":
    set_name = general_keys.VALIDATION
    fontsize = 12
    results_folder_name = "final_hyperparam_search"
    results_folder_path = os.path.join(PROJECT_PATH, "results", results_folder_name)

    results_path = os.path.join(results_folder_path, "%s_set_results.pkl" % set_name)
    results = pd.read_pickle(results_path)
    # pprint(results)
    # print(len(list(results.keys())))

    names = []
    means = []
    stds = []

    for key in results.keys():
        names.append(key)
        means.append(results[key]["mean"])
        stds.append(results[key]["std"])

    sorted_indxs = np.argsort(means)
    means = np.array(means)[sorted_indxs]
    names = np.array(names)[sorted_indxs]
    stds = np.array(stds)[sorted_indxs]

    for i in range(len(means)):
        print("%s %.3f +/- %.3f" % (names[i], means[i], stds[i]))

    colors = np.random.RandomState(12).rand(len(means), 3)
    for i in range(len(means)):
        plt.scatter(stds[i], means[i], c=colors[i], alpha=0.8)
    plt.title(
        "%s set accuracy for %i models from\n random hyperparameter search"
        % (set_name, len(means))
    )
    plt.ylabel("Accuracy", fontsize=fontsize)
    plt.grid(True, linestyle="--")
    plt.xlabel("Accuracy's standar deviation", fontsize=fontsize)
    plt.savefig(
        os.path.join(results_folder_path, "All_models.pdf"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.show()

    n_model_to_show = 5
    n_model_means = means[int(-1 * n_model_to_show) :]
    n_model_colors = colors[int(-1 * n_model_to_show) :]
    n_model_stds = stds[int(-1 * n_model_to_show) :]
    n_model_names = names[int(-1 * n_model_to_show) :]
    n_model_sorted_stds = np.sort(n_model_stds)

    delta_percentage = 0.1
    delta_mean = (n_model_means.max() - n_model_means.min()) * delta_percentage
    delta_std = (n_model_stds.max() - n_model_stds.min()) * delta_percentage

    plt.figure(figsize=(10, 6))
    for i in range(len(n_model_means)):
        i = n_model_to_show - i - 1
        print(i)
        model_param_dict = parse_acronym_to_param_dict_abbreviation(n_model_names[i])
        model_name_str = parse_param_dict_to_str(model_param_dict)
        model_name_str = r"$M_{%i}$: " % (n_model_to_show - i - 1) + model_name_str
        plt.scatter(
            n_model_stds[i],
            n_model_means[i],
            # c=n_model_colors[i],
            label=model_name_str,
        )
        plt.errorbar(
            n_model_stds[i], n_model_means[i], yerr=n_model_stds[i]
        )  # , c=n_model_colors[i])
    plt.xlim([n_model_stds.min() - delta_std, n_model_sorted_stds.max() + delta_std])
    plt.ylabel("Accuracy", fontsize=fontsize)
    plt.xlabel("Accuracy's standar deviation", fontsize=fontsize)
    # plt.ylim(n_model_means.min()-delta_mean, n_model_means.max()+delta_mean)
    # plt.legend(loc='upper left')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(True, linestyle="--")
    plt.tight_layout(rect=[0, 0, 0.75, 1])
    plt.savefig(
        os.path.join(results_folder_path, "Best_models.pdf"),
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.show()

    # print(list(zip(n_model_means, n_model_stds)))
    # printing models
    for single_name in n_model_names:
        print(single_name, ":", results[single_name])
    # print(len(list(zip(n_model_means, n_model_stds))))

    # getting welch tests
    # worst_best_dict = eval("{'DeepHits_EntropyRegBeta0.5000_batch32_lr0.00010"
    #                        "_droprate0.5000_inputsize21_filtersize3': [0.87, "
    #                        "0.88, 0.88, 0.8733333333333333, 0.8566666666666667],"
    #                        " 'DeepHits_EntropyRegBeta0.8000_batch64_lr0.00005"
    #                        "_droprate0.8000_inputsize41_filtersize5': [0.87, "
    #                        "0.87, 0.89, 0.86, 0.8766666666666667]}")
    worst_best_dict = {
        n_model_names[-1]: results[n_model_names[-1]]["metric_values"],
        n_model_names[0]: results[n_model_names[0]]["metric_values"],
    }
    print(worst_best_dict)
    welchs_t_test = stats.ttest_ind(
        list(worst_best_dict.values())[0],
        list(worst_best_dict.values())[1],
        equal_var=False,
    )[1]
    print(welchs_t_test)

    # getting welch tests time
    # worst_best_dict = eval("{'DeepHits_EntropyRegBeta0.5000_batch32_lr0.00010_"
    #                        "droprate0.5000_inputsize21_filtersize3': "
    #                        "[0.023091793060302734, 0.018239498138427734, "
    #                        "0.01804351806640625, 0.018353700637817383, "
    #                        "0.01821160316467285],"
    #                        "'DeepHits_EntropyRegBeta0.8000_batch64_lr0.00005_"
    #                        "droprate0.8000_inputsize41_filtersize5': "
    #                        "[0.02687692642211914, 0.02144932746887207, "
    #                        "0.019466161727905273, 0.02012181282043457, "
    #                        "0.019135713577270508]}")
    worst_best_dict = {
        n_model_names[-1]: results[n_model_names[-1]]["time_values"],
        n_model_names[0]: results[n_model_names[0]]["time_values"],
    }
    print(worst_best_dict)
    welchs_t_test = stats.ttest_ind(
        list(worst_best_dict.values())[0],
        list(worst_best_dict.values())[1],
        equal_var=False,
    )[1]
    print(welchs_t_test)
