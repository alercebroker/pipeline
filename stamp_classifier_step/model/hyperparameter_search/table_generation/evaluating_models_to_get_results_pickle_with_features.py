import os
import sys

"""
Run this to generate results pickle in order to plot result with plot_results_pickle.py
final_hyperparam_search folder must contain all model folders, without any nestings
"""

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)

from hyperparameter_search.table_generation.imposing_folder_and_trainings_consistency import (
    removing_trains_to_have_only_N_trains,
    get_folder_names_in_path,
    removing_incomplete_trains,
    count_unique_folders_names,
    parse_acronym_to_param_dict,
    change_learning_rate_names_in_results,
)
from models.classifiers.deepHits_with_features_entropy_reg import (
    DeepHiTSWithFeaturesEntropyReg,
)
from parameters import param_keys, general_keys
import numpy as np
import pickle
import time
from modules.data_loaders.frame_to_input_with_features import FrameToInputWithFeatures
from tqdm import tqdm
from natsort import natsorted


def get_accuracy(labels, predictions):
    return np.mean(labels == predictions)


def calculate_metric(labels, predictions):
    return get_accuracy(labels, predictions)


def delta_timer(delta_time):
    hours, rem = divmod(delta_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def evaluate_all_models(results_folder_path, dataset_to_evaluate, params, set_name):
    # get acronims
    all_trained_model_folder_name = get_folder_names_in_path(results_folder_path)
    folder_acronyms, _ = count_unique_folders_names(all_trained_model_folder_name)
    # print(folder_acronyms)
    results_dict = {}
    execution_times_dict = {}
    # evaluate models of each acronym
    for acronym in tqdm(folder_acronyms):
        # if acronym == 'DeepHits_EntropyRegBeta0.8000_batch64_lr0.00005_droprate0.8000_inputsize41_filtersize5' or\
        #     acronym == 'DeepHits_EntropyRegBeta0.8000_batch16_lr0.00010_droprate0.8000_inputsize41_filtersize5' or \
        #     acronym == 'DeepHits_EntropyRegBeta0.8000_batch64_lr0.00010_droprate0.5000_inputsize21_filtersize3' or \
        #     acronym == 'DeepHits_EntropyRegBeta1.0000_batch64_lr0.00010_droprate0.2000_inputsize41_filtersize3' or \
        #   acronym == 'DeepHits_EntropyRegBeta0.5000_batch32_lr0.00010_droprate0.5000_inputsize21_filtersize3':
        #   abc=None
        # else:
        #   continue
        acronym_params = parse_acronym_to_param_dict(acronym)
        params.update(acronym_params)
        model = DeepHiTSWithFeaturesEntropyReg(params)
        train_set, val_set, test_set = model._prepare_input()
        aux_dict_data = {
            general_keys.TRAIN: train_set,
            general_keys.VALIDATION: val_set,
            general_keys.TEST: test_set,
        }
        print("Metadata no norm %f" % np.mean(dataset_to_evaluate.meta_data))
        preprocessed_data_set = model.preprocess_data(
            dataset_to_evaluate.data_array, dataset_to_evaluate.meta_data
        )
        print("Metadata norm %f" % np.mean(preprocessed_data_set.meta_data))
        data_label = dataset_to_evaluate.data_label
        results_dict[acronym] = []
        execution_times_dict[acronym] = []
        folder_names_that_match_acronym = [
            s for s in all_trained_model_folder_name if any(xs in s for xs in [acronym])
        ]
        print(folder_names_that_match_acronym)
        folder_names_that_match_acronym = natsorted(
            folder_names_that_match_acronym, key=lambda y: y.lower()
        )
        print(folder_names_that_match_acronym)
        for folder_name in folder_names_that_match_acronym:
            weights_path = os.path.join(
                results_folder_path, folder_name, "checkpoints", "model"
            )
            model.load_model(weights_path)
            print(
                "%s Meta_data equal: %i"
                % (
                    set_name,
                    int(
                        np.mean(
                            aux_dict_data[set_name].meta_data
                            == preprocessed_data_set.meta_data
                        )
                    ),
                )
            )
            print(
                "%s images equal: %i"
                % (
                    set_name,
                    int(
                        np.mean(
                            aux_dict_data[set_name].data_array
                            == preprocessed_data_set.data_array
                        )
                    ),
                )
            )
            predictions = model.predict(
                preprocessed_data_set.data_array, preprocessed_data_set.meta_data
            )
            metric_value = calculate_metric(data_label, predictions)
            results_dict[acronym].append(metric_value)
            # measure inference time
            sample = dataset_to_evaluate.data_array[0][None, ...]
            sample_meta_data = dataset_to_evaluate.meta_data[0][None, ...]
            # preprocessed_sample = model.preprocess_data_array(sample)
            # model.predict(preprocessed_sample)
            start = time.time()
            preprocessed_single_dataset = model.preprocess_data(
                sample, sample_meta_data
            )
            model.predict(
                preprocessed_single_dataset.data_array,
                preprocessed_single_dataset.meta_data,
            )
            end = time.time()
            execution_times_dict[acronym].append(end - start)
        model.close()
        del model
        print(
            "\nRESULTS: ",
            acronym,
            " ",
            np.mean(results_dict[acronym]),
            ":",
            results_dict[acronym],
        )
    # print(execution_times_dict)
    return results_dict, execution_times_dict


def get_mean_dict(results_dict: dict) -> dict:
    mean_results_dict = {}
    for key in results_dict.keys():
        mean_results_dict[key] = np.mean(results_dict[key])
    return mean_results_dict


def get_std_dict(results_dict: dict) -> dict:
    std_results_dict = {}
    for key in results_dict.keys():
        std_results_dict[key] = np.std(results_dict[key])
    return std_results_dict


def main(set_name):
    results_folder_name = "final_hyperparam_search_v2"
    results_folder_path = os.path.join(PROJECT_PATH, "results", results_folder_name)

    # cleaning unnecessary folders
    removing_incomplete_trains(results_folder_path)
    change_learning_rate_names_in_results(results_folder_path)
    removing_trains_to_have_only_N_trains(results_folder_path)
    all_trained_model_folder_name = get_folder_names_in_path(results_folder_path)
    _, counts_in_acronyms = count_unique_folders_names(all_trained_model_folder_name)
    assert np.sum(counts_in_acronyms) % 5 == 0

    # data to evaluate
    data_path = os.path.join(PROJECT_PATH, "../pickles", "converted_data.pkl")
    # 'training_set_Nov-26-2019.pkl')
    n_classes = 5
    params = {
        param_keys.RESULTS_FOLDER_NAME: "aux_model",
        param_keys.DATA_PATH_TRAIN: data_path,
        param_keys.WAIT_FIRST_EPOCH: False,
        param_keys.N_INPUT_CHANNELS: 3,
        param_keys.CHANNELS_TO_USE: [0, 1, 2],
        param_keys.TRAIN_ITERATIONS_HORIZON: 30000,
        param_keys.TRAIN_HORIZON_INCREMENT: 10000,
        param_keys.TEST_SIZE: n_classes * 100,
        param_keys.VAL_SIZE: n_classes * 100,
        param_keys.NANS_TO: 0,
        param_keys.NUMBER_OF_CLASSES: n_classes,
        param_keys.CROP_SIZE: 63,
        param_keys.INPUT_IMAGE_SIZE: 63,
        param_keys.LEARNING_RATE: 0.0001,
        param_keys.VALIDATION_MONITOR: general_keys.LOSS,
        param_keys.VALIDATION_MODE: general_keys.MIN,
        param_keys.ENTROPY_REG_BETA: 0.5,
        param_keys.DROP_RATE: 0.5,
        param_keys.BATCH_SIZE: 32,
        param_keys.KERNEL_SIZE: 3,
        param_keys.FEATURES_NAMES_LIST: [
            "sgscore1",
            "distpsnr1",
            "sgscore2",
            "distpsnr2",
            "sgscore3",
            "distpsnr3",
            "isdiffpos",
            "fwhm",
            "magpsf",
            "sigmapsf",
            "ra",
            "dec",
            "diffmaglim",
            "classtar",
            "ndethist",
            "ncovhist",
            "ecl_lat",
            "ecl_long",
            "gal_lat",
            "gal_long",
            "non_detections",
            "chinr",
            "sharpnr",
        ],
        param_keys.BATCHNORM_FEATURES_FC: True,
        param_keys.FEATURES_CLIPPING_DICT: {
            "sgscore1": [-1, "max"],
            "distpsnr1": [-1, "max"],
            "sgscore2": [-1, "max"],
            "distpsnr2": [-1, "max"],
            "sgscore3": [-1, "max"],
            "distpsnr3": [-1, "max"],
            "fwhm": ["min", 10],
            "distnr": [-1, "max"],
            "magnr": [-1, "max"],
            "ndethist": ["min", 20],
            "ncovhist": ["min", 3000],
            "chinr": [-1, 15],
            "sharpnr": [-1, 1.5],
            "non_detections": ["min", 2000],
        },
    }
    aux_model = DeepHiTSWithFeaturesEntropyReg(params)
    params = aux_model.params
    aux_data_loader = FrameToInputWithFeatures(params)
    aux_data_loader.dataset_preprocessor.set_pipeline(
        [
            aux_data_loader.dataset_preprocessor.image_check_single_image,
            aux_data_loader.dataset_preprocessor.image_clean_misshaped,
            aux_data_loader.dataset_preprocessor.image_select_channels,
        ]
    )
    dataset_dicts = aux_data_loader.get_preprocessed_datasets_splitted()
    set_to_evaluate = dataset_dicts[set_name]
    aux_model.close()
    del aux_model
    del aux_data_loader
    del dataset_dicts
    # print(train_set.data_array.shape, val_set.data_array.shape,
    #       test_set.data_array.shape)

    results_dict, execution_times_dict = evaluate_all_models(
        results_folder_path, set_to_evaluate, params, set_name
    )
    # results_dict = {'DeepHits_EntropyRegBeta0.0000_batch16_lr0.00500_droprate0.5000_inputsize41_filtersize7': [0.2, 0.2, 0.2, 0.19666666666666666, 0.2], 'DeepHits_EntropyRegBeta0.0000_batch32_lr0.00010_droprate0.2000_inputsize41_filtersize5': [0.8466666666666667, 0.85, 0.8633333333333333, 0.86, 0.8533333333333334], 'DeepHits_EntropyRegBeta0.0000_batch32_lr0.00500_droprate0.2000_inputsize63_filtersize3': [0.37, 0.7733333333333333, 0.78, 0.8133333333333334, 0.7633333333333333], 'DeepHits_EntropyRegBeta0.0000_batch32_lr0.00500_droprate0.2000_inputsize63_filtersize7': [0.2, 0.79, 0.7366666666666667, 0.7066666666666667, 0.2], 'DeepHits_EntropyRegBeta0.0000_batch64_lr0.00100_droprate0.5000_inputsize41_filtersize3': [0.83, 0.8333333333333334, 0.8333333333333334, 0.85, 0.8633333333333333], 'DeepHits_EntropyRegBeta0.3000_batch16_lr0.00010_droprate0.5000_inputsize63_filtersize7': [0.8233333333333334, 0.8566666666666667, 0.8533333333333334, 0.8533333333333334, 0.8333333333333334], 'DeepHits_EntropyRegBeta0.3000_batch16_lr0.00100_droprate0.8000_inputsize21_filtersize5': [0.8166666666666667, 0.86, 0.83, 0.84, 0.8566666666666667], 'DeepHits_EntropyRegBeta0.3000_batch32_lr0.00010_droprate0.2000_inputsize41_filtersize5': [0.8633333333333333, 0.8666666666666667, 0.8533333333333334, 0.8633333333333333, 0.86], 'DeepHits_EntropyRegBeta0.3000_batch32_lr0.00010_droprate0.8000_inputsize63_filtersize3': [0.8433333333333334, 0.85, 0.8366666666666667, 0.8366666666666667, 0.8366666666666667], 'DeepHits_EntropyRegBeta0.5000_batch16_lr0.00010_droprate0.8000_inputsize63_filtersize7': [0.8533333333333334, 0.85, 0.8533333333333334, 0.86, 0.8566666666666667], 'DeepHits_EntropyRegBeta0.5000_batch16_lr0.00500_droprate0.8000_inputsize63_filtersize7': [0.2, 0.2, 0.2, 0.2, 0.2], 'DeepHits_EntropyRegBeta0.5000_batch32_lr0.00500_droprate0.2000_inputsize21_filtersize3': [0.2, 0.8366666666666667, 0.8466666666666667, 0.2, 0.83], 'DeepHits_EntropyRegBeta0.5000_batch64_lr0.00010_droprate0.2000_inputsize41_filtersize5': [0.8533333333333334, 0.86, 0.8766666666666667, 0.87, 0.8733333333333333], 'DeepHits_EntropyRegBeta0.5000_batch64_lr0.00010_droprate0.2000_inputsize63_filtersize3': [0.8566666666666667, 0.8533333333333334, 0.8533333333333334, 0.8566666666666667, 0.8433333333333334], 'DeepHits_EntropyRegBeta0.5000_batch64_lr0.00010_droprate0.8000_inputsize21_filtersize7': [0.8633333333333333, 0.85, 0.86, 0.8566666666666667, 0.8666666666666667], 'DeepHits_EntropyRegBeta0.5000_batch64_lr0.00010_droprate0.8000_inputsize41_filtersize3': [0.8733333333333333, 0.87, 0.8666666666666667, 0.86, 0.8566666666666667], 'DeepHits_EntropyRegBeta0.8000_batch16_lr0.00100_droprate0.2000_inputsize21_filtersize5': [0.82, 0.8433333333333334, 0.8633333333333333, 0.8233333333333334, 0.86], 'DeepHits_EntropyRegBeta0.8000_batch32_lr0.00010_droprate0.5000_inputsize63_filtersize7': [0.8533333333333334, 0.85, 0.87, 0.86, 0.8433333333333334], 'DeepHits_EntropyRegBeta0.8000_batch64_lr0.00500_droprate0.2000_inputsize41_filtersize5': [0.8133333333333334, 0.7966666666666666, 0.81, 0.8266666666666667, 0.8133333333333334], 'DeepHits_EntropyRegBeta0.8000_batch64_lr0.00500_droprate0.5000_inputsize21_filtersize5': [0.8166666666666667, 0.2, 0.82, 0.8033333333333333, 0.2], 'DeepHits_EntropyRegBeta1.0000_batch16_lr0.00010_droprate0.2000_inputsize41_filtersize7': [0.8633333333333333, 0.8633333333333333, 0.87, 0.8633333333333333, 0.86], 'DeepHits_EntropyRegBeta1.0000_batch32_lr0.00010_droprate0.8000_inputsize21_filtersize5': [0.8633333333333333, 0.8733333333333333, 0.8533333333333334, 0.85, 0.8666666666666667], 'DeepHits_EntropyRegBeta1.0000_batch32_lr0.00010_droprate0.8000_inputsize41_filtersize7': [0.87, 0.8533333333333334, 0.8666666666666667, 0.8866666666666667, 0.8566666666666667], 'DeepHits_EntropyRegBeta1.0000_batch64_lr0.00050_droprate0.5000_inputsize63_filtersize5': [0.8566666666666667, 0.86, 0.8666666666666667, 0.8433333333333334, 0.8566666666666667], 'DeepHits_EntropyRegBeta1.0000_batch64_lr0.00500_droprate0.5000_inputsize63_filtersize7': [0.7733333333333333, 0.2, 0.2, 0.2, 0.2]}
    mean_results = get_mean_dict(results_dict)
    std_results = get_std_dict(results_dict)
    mean_time_results = get_mean_dict(execution_times_dict)
    std_time_results = get_std_dict(execution_times_dict)

    keys = list(mean_results.keys())
    values = list(results_dict.values())
    time_values = list(execution_times_dict.values())
    means = list(mean_results.values())
    stds = list(std_results.values())
    time_means = list(mean_time_results.values())
    time_stds = list(std_time_results.values())
    sorting_indixes = np.argsort(means)
    # print(sorting_indixes)
    mean_std_results = {}
    for sort_i in sorting_indixes:
        mean_std_results[keys[sort_i]] = {
            "mean": means[sort_i],
            "std": stds[sort_i],
            "metric_values": values[sort_i],
            "time_mean": time_means[sort_i],
            "time_std": time_stds[sort_i],
            "time_values": time_values[sort_i],
        }
        print("%s: %s" % (keys[sort_i], str(mean_std_results[keys[sort_i]])))

    print(len(list(mean_std_results.keys())))

    save_path = os.path.join(results_folder_path, "%s_set_results.pkl" % set_name)
    with open(save_path, "wb") as handle:
        pickle.dump(mean_std_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main(general_keys.TEST)
    main(general_keys.VALIDATION)
