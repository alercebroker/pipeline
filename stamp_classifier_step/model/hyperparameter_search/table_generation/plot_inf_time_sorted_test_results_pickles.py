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


def get_names_means_std_sorted(results_dict, sorted_indxs=None):
    names = []
    means = []
    stds = []
    time_means = []
    time_stds = []
    for key in results_dict.keys():
        names.append(key)
        means.append(results_dict[key]["mean"])
        stds.append(results_dict[key]["std"])
        time_means.append(results_dict[key]["time_mean"])
        time_stds.append(results_dict[key]["time_std"])

    if sorted_indxs is None:
        means_str = ["%.16f" % x for x in means]
        stds_str = ["%.16f" % (1 - x) for x in stds]
        concat_str = [m + n for m, n in zip(means_str, stds_str)]
        sorted_indxs = np.argsort(concat_str)
    else:
        sorted_indxs = np.argsort(means)
    means = np.array(means)[sorted_indxs]
    names = np.array(names)[sorted_indxs]
    stds = np.array(stds)[sorted_indxs]
    time_stds = np.array(time_stds)[sorted_indxs] * 1000
    time_means = np.array(time_means)[sorted_indxs] * 1000
    return names, means, stds, time_means, time_stds, sorted_indxs


def get_welchs_and_print_best_worst(result_dict, n_model_names):
    # print(list(zip(n_model_means, n_model_stds)))
    # printing models
    for single_name in n_model_names:
        print(single_name, ":", result_dict[single_name])
    # print(len(list(zip(n_model_means, n_model_stds))))

    # getting welch tests
    # worst_best_dict = eval("{'DeepHits_EntropyRegBeta0.5000_batch32_lr0.00010"
    #                        "_droprate0.5000_inputsize21_filtersize3': [0.87, "
    #                        "0.88, 0.88, 0.8733333333333333, 0.8566666666666667],"
    #                        " 'DeepHits_EntropyRegBeta0.8000_batch64_lr0.00005"
    #                        "_droprate0.8000_inputsize41_filtersize5': [0.87, "
    #                        "0.87, 0.89, 0.86, 0.8766666666666667]}")
    worst_best_dict = {
        n_model_names[-1]: result_dict[n_model_names[-1]]["metric_values"],
        n_model_names[0]: result_dict[n_model_names[0]]["metric_values"],
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
        n_model_names[-1]: result_dict[n_model_names[-1]]["time_values"],
        n_model_names[0]: result_dict[n_model_names[0]]["time_values"],
    }
    print(worst_best_dict)
    welchs_t_test = stats.ttest_ind(
        list(worst_best_dict.values())[0],
        list(worst_best_dict.values())[1],
        equal_var=False,
    )[1]
    print(welchs_t_test)


if __name__ == "__main__":
    color_random_state = 63  # np.random.randint(1000)#918#
    print(color_random_state)
    set_name = general_keys.TEST
    val_set_name = general_keys.VALIDATION
    fontsize = 12
    results_folder_name = "final_hyperparam_search_v1"
    results_folder_path = os.path.join(PROJECT_PATH, "results", results_folder_name)

    results_path = os.path.join(
        results_folder_path, "%s_set_results_with_esp_cases.pkl" % set_name
    )
    test_results = pd.read_pickle(results_path)

    val_results_path = os.path.join(
        results_folder_path, "%s_set_results_with_esp_cases.pkl" % val_set_name
    )
    val_results = pd.read_pickle(val_results_path)
    # pprint(results)
    # print(len(list(results.keys())))

    (
        val_names,
        val_means,
        val_stds,
        val_time_means,
        val_time_stds,
        val_sorted_idxs,
    ) = get_names_means_std_sorted(val_results)

    (
        test_names,
        test_means,
        test_stds,
        test_time_means,
        test_time_stds,
        _,
    ) = get_names_means_std_sorted(test_results, False)
    colors = np.random.RandomState(color_random_state).rand(len(test_means), 3)

    n_model_to_show = 5
    val_n_model_means = val_means[int(-1 * n_model_to_show) :]
    val_n_model_stds = val_stds[int(-1 * n_model_to_show) :]
    val_n_model_names = val_names[int(-1 * n_model_to_show) :]

    # getting test from best vals
    test_n_model_means = []
    test_n_model_stds = []
    test_n_model_names = []
    test_n_model_time_means = []
    test_n_model_time_stds = []
    for i in range(n_model_to_show):
        val_model_name_i = val_n_model_names[i]
        test_idx_with_same_name_as_val = np.where(test_names == val_model_name_i)[0][0]
        test_std_i = test_stds[test_idx_with_same_name_as_val]
        test_n_model_stds.append(test_std_i)
        test_mean_i = test_means[test_idx_with_same_name_as_val]
        test_n_model_means.append(test_mean_i)
        test_name_i = test_names[test_idx_with_same_name_as_val]
        test_n_model_names.append(test_name_i)

        test_time_std_i = test_time_stds[test_idx_with_same_name_as_val]
        test_n_model_time_stds.append(test_time_std_i)
        test_time_mean_i = test_time_means[test_idx_with_same_name_as_val]
        test_n_model_time_means.append(test_time_mean_i)
        print("NAMES ARE EQUAL?: %i" % int(test_name_i == val_model_name_i))
        test_color_i = colors[test_idx_with_same_name_as_val]
    test_n_model_names = np.array(test_n_model_names)
    test_n_model_means = np.array(test_n_model_means)
    test_n_model_stds = np.array(test_n_model_stds)
    test_n_model_time_means = np.array(test_n_model_time_means)
    test_n_model_time_stds = np.array(test_n_model_time_stds)

    sorted_indxs = np.argsort(test_n_model_means)
    test_n_model_names = np.array(test_n_model_names)[sorted_indxs]
    test_n_model_means = np.array(test_n_model_means)[sorted_indxs]
    test_n_model_stds = np.array(test_n_model_stds)[sorted_indxs]
    test_n_model_time_means = np.array(test_n_model_time_means)[sorted_indxs]
    test_n_model_time_stds = np.array(test_n_model_time_stds)[sorted_indxs]

    # plotting test
    fig = plt.figure(figsize=(6.4, 4.8))
    for i in range(n_model_to_show):
        i = n_model_to_show - i - 1
        print(i)
        model_param_dict = parse_acronym_to_param_dict_abbreviation(
            test_n_model_names[i]
        )
        model_name_str = parse_param_dict_to_str(model_param_dict)
        model_name_str = r"$M_{%i}$: " % (n_model_to_show - i - 1) + model_name_str
        test_model_name_i = test_n_model_names[i]
        test_idx_with_same_name_as_val = np.where(test_names == test_model_name_i)[0][0]
        test_color_i = colors[test_idx_with_same_name_as_val]
        plt.errorbar(
            test_n_model_time_means[i],
            test_n_model_means[i],
            c=test_color_i,
            yerr=test_n_model_stds[i],
            xerr=test_n_model_time_stds[i],
        )
        plt.scatter(
            test_n_model_time_means[i],
            test_n_model_means[i],
            c=test_color_i,
            marker="D",
            s=50,
            edgecolor="black",
            label=model_name_str,
        )

    delta_percentage = 0.1
    test_delta_mean = (
        test_n_model_means.max() - test_n_model_means.min()
    ) * delta_percentage
    test_delta_time = (
        test_n_model_time_means.max() - test_n_model_time_means.min()
    ) * delta_percentage

    # plt.xlim(
    #     [test_n_model_stds.min() - test_delta_time,
    #      test_n_model_stds.max() + test_delta_time])
    plt.ylabel("Test accuracy", fontsize=fontsize)
    plt.xlabel("Inference time [ms]", fontsize=fontsize)
    # plt.ylim(n_model_means.min()-delta_mean, n_model_means.max()+delta_mean)
    # plt.legend(loc='upper left')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.grid(True, linestyle="--")
    # plt.tight_layout(rect=[0, 0, 0.75, 1])
    fig.tight_layout(pad=0.2)
    plt.savefig(os.path.join(results_folder_path, "Best_models.pdf"))  # ,
    # bbox_inches='tight')#,
    # pad_inches=0)
    plt.show()

    print("\nWELCHS VAL")
    get_welchs_and_print_best_worst(val_results, val_n_model_names)

    print("\nWELCHS TEST")
    get_welchs_and_print_best_worst(test_results, test_n_model_names)

    names = []
    means = []
    time_means = []

    for key in test_results.keys():
        names.append(key)
        means.append(test_results[key]["mean"])
        time_means.append(test_results[key]["time_mean"])

    sorted_indxs = np.argsort(means)
    means = np.array(means)[sorted_indxs]
    names = np.array(names)[sorted_indxs]
    time_means = np.array(time_means)[sorted_indxs] * 1000

    # for i in range(len(means)):
    #   print('%s %.3f +/- %.3f' % (names[i], means[i], stds[i]))

    fig = plt.figure(figsize=(6.4, 4.8))
    for i in range(len(means)):
        if names[i] in test_n_model_names:
            plt.scatter(
                time_means[i],
                means[i],
                marker="D",
                s=50,
                c=colors[i],
                alpha=1,
                edgecolor="black",
            )
        else:
            plt.scatter(time_means[i], means[i], marker="o", c=colors[i], alpha=0.7)
    plt.title(
        "Test set accuracy for %i models from\n random hyperparameter search"
        % len(means)
    )
    plt.ylabel("Test accuracy", fontsize=fontsize)
    plt.grid(True, linestyle="--")
    plt.xlabel("Inference time [ms]", fontsize=fontsize)
    delta_percentage = 0.1
    test_delta_mean = (means.max() - means.min()) * delta_percentage
    test_delta_time = (time_means.max() - time_means.min()) * delta_percentage

    plt.xlim([time_means.min() - test_delta_time, time_means.max() + test_delta_time])
    plt.ylim([means.min() - test_delta_mean, means.max() + test_delta_mean])
    fig.tight_layout(pad=0.2)
    plt.savefig(os.path.join(results_folder_path, "All_models.pdf"))  # ,
    # bbox_inches='tight',
    # pad_inches=0)
    plt.show()
