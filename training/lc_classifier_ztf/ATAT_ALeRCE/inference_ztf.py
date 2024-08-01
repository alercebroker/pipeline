import glob
import h5py
import torch
import json

from joblib import load

import importlib
import yaml
import tqdm

import numpy as np
import copy

import warnings

warnings.filterwarnings("ignore")

import sys
import os

from training import load_yaml

# os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-98b55758-f7a3-59db-8607-5be6d2eeb06b"
# os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-802fb5a9-d98a-5bce-ad53-9227189dc2cf"


def obtain_valid_mask(sample, mask, time_alert, eval_time):
    mask_time = time_alert <= eval_time
    sample["mask"] = mask * mask_time
    return sample


def get_tabular_data(feat_col, dict_add_feat_col, config_used, eval_time):
    new_feat_col = copy.deepcopy(feat_col)

    if config_used["using_metadata"]:

        if not config_used["not_quantile_transformer"]:
            metadata_qt = load(
                "./final_dataset/QT-New/md_fold_{}.joblib".format(config_used["seed"])
            )
            new_feat_col = torch.from_numpy(metadata_qt.transform(feat_col))

        if config_used["using_features"]:
            add_feat_col = dict_add_feat_col["{}".format(eval_time)]

            if config_used["not_quantile_transformer"]:
                new_add_feat_col = copy.deepcopy(add_feat_col)

            else:
                features_qt = load(
                    "./final_dataset/QT-New/fe_2048_fold_{}.joblib".format(
                        config_used["seed"]
                    )
                )
                new_add_feat_col = torch.from_numpy(features_qt.transform(add_feat_col))

            new_feat_col_total = torch.cat([new_feat_col, new_add_feat_col], dim=1)

        else:
            new_feat_col_total = copy.deepcopy(new_feat_col)

    else:
        if config_used["using_features"]:
            add_feat_col = dict_add_feat_col["{}".format(eval_time)]

            if config_used["not_quantile_transformer"]:
                new_feat_col_total = copy.deepcopy(add_feat_col)
            else:
                features_qt = load(
                    "./final_dataset/QT-New/fe_2048_fold_{}.joblib".format(
                        config_used["seed"]
                    )
                )
                new_feat_col_total = torch.from_numpy(
                    features_qt.transform(add_feat_col)
                )

        else:
            new_feat_col_total = copy.deepcopy(new_feat_col)

    return new_feat_col_total


def get_chunks(data_dict_eval_time, eval_time, batch_size, config_used):
    data_dict_eval_time["test_{}".format(eval_time)]["data"] = [
        data_dict_eval_time["test_{}".format(eval_time)]["data"][x : x + batch_size]
        for x in range(
            0, len(data_dict_eval_time["test_{}".format(eval_time)]["data"]), batch_size
        )
    ]
    data_dict_eval_time["test_{}".format(eval_time)]["data_err"] = [
        data_dict_eval_time["test_{}".format(eval_time)]["data_err"][x : x + batch_size]
        for x in range(
            0,
            len(data_dict_eval_time["test_{}".format(eval_time)]["data_err"]),
            batch_size,
        )
    ]
    data_dict_eval_time["test_{}".format(eval_time)]["time"] = [
        data_dict_eval_time["test_{}".format(eval_time)]["time"][x : x + batch_size]
        for x in range(
            0, len(data_dict_eval_time["test_{}".format(eval_time)]["time"]), batch_size
        )
    ]
    data_dict_eval_time["test_{}".format(eval_time)]["mask"] = [
        data_dict_eval_time["test_{}".format(eval_time)]["mask"][x : x + batch_size]
        for x in range(
            0, len(data_dict_eval_time["test_{}".format(eval_time)]["mask"]), batch_size
        )
    ]
    data_dict_eval_time["test_{}".format(eval_time)]["labels"] = [
        data_dict_eval_time["test_{}".format(eval_time)]["labels"][x : x + batch_size]
        for x in range(
            0,
            len(data_dict_eval_time["test_{}".format(eval_time)]["labels"]),
            batch_size,
        )
    ]

    if config_used["general"]["use_metadata"]:
        data_dict_eval_time["test_{}".format(eval_time)]["metadata_feat"] = [
            data_dict_eval_time["test_{}".format(eval_time)]["metadata_feat"][
                x : x + batch_size
            ]
            for x in range(
                0,
                len(data_dict_eval_time["test_{}".format(eval_time)]["metadata_feat"]),
                batch_size,
            )
        ]

    if config_used["general"]["use_features"]:
        data_dict_eval_time["test_{}".format(eval_time)]["extracted_feat"] = [
            data_dict_eval_time["test_{}".format(eval_time)]["extracted_feat"][
                x : x + batch_size
            ]
            for x in range(
                0,
                len(data_dict_eval_time["test_{}".format(eval_time)]["extracted_feat"]),
                batch_size,
            )
        ]

    # data_dict_eval_time['test_{}'.format(eval_time)]['tabular_feat'] = \
    #    [data_dict_eval_time['test_{}'.format(eval_time)]['tabular_feat'][x:x+batch_size] for x in range(0, len(data_dict_eval_time['test_{}'.format(eval_time)]['tabular_feat']), batch_size)]

    return data_dict_eval_time


from collections import OrderedDict
from src.layers import ATAT


def get_predictions(
    path_exp, data_root, list_eval_time, batch_size, device, partition_used
):

    config_used = load_yaml("./{}/args.yaml".format(path_exp))
    # config_used = load('./{}/args.pickle'.format(path_exp))

    print("Loading data ...")
    h5_file = h5py.File("{}/dataset.h5".format(data_root))

    # these_idx = h5_file.get("test")
    test_idx = h5_file.get("test")[:]
    validation_idx = h5_file.get("validation_0")[:]
    training_idx = h5_file.get("training_0")[:]
    these_idx = np.concatenate((test_idx, validation_idx, training_idx))
    sorted_idx = np.argsort(these_idx)
    these_idx = these_idx[sorted_idx]
    print("Number of indices: {}".format(len(these_idx)))

    SNID = h5_file.get("SNID")[these_idx]
    data = h5_file.get("flux")[these_idx]
    data_err = h5_file.get("flux_err")[these_idx]
    mask = h5_file.get("mask")[these_idx]
    time = h5_file.get("time")[these_idx]
    target = h5_file.get("labels")[these_idx]

    # print('- loading calculated features ...')
    # dict_add_feat_col = dict()
    # for eval_time in list_eval_time:
    #    dict_add_feat_col['{}'.format(eval_time)] = torch.from_numpy(h5_file.get('norm_add_feat_col_{}'.format(eval_time))[:][these_idx])

    data_dict = {
        "data": torch.from_numpy(data).float(),
        "data_err": torch.from_numpy(data_err).float(),
        "time": torch.from_numpy(time).float(),
        "mask": torch.from_numpy(mask).float(),
        "labels": torch.from_numpy(target).long(),
    }

    if config_used["general"]["use_metadata"]:
        metadata_feat = h5_file.get("metadata_feat")[:][these_idx]
        path_QT = "./{}/quantiles/metadata/fold_{}.joblib".format(
            data_root, partition_used
        )

        if config_used["general"]["use_QT"]:
            QT = load(path_QT)
            metadata_feat = QT.transform(metadata_feat)

        data_dict.update(
            {"metadata_feat": torch.from_numpy(metadata_feat).float().unsqueeze(2)}
        )

    if config_used["general"]["use_features"]:
        extracted_feat = dict()
        for time_eval in config_used["general"]["list_time_to_eval"]:
            path_QT = f"./{data_root}/quantiles/features/fold_{partition_used}.joblib"
            extracted_feat_aux = h5_file.get("extracted_feat_{}".format(time_eval))[:][
                these_idx
            ]
            if config_used["general"]["use_QT"]:
                QT = load(path_QT)
                extracted_feat_aux = QT.transform(extracted_feat_aux)

            extracted_feat.update({time_eval: extracted_feat_aux})

        # data_dict.update({"extracted_feat": torch.from_numpy(extracted_feat).float().unsqueeze(2)})

    h5_file.close()

    ##########################################################################
    print("\nLoading model ...")
    print("path_exp: {}".format(path_exp))

    model = ATAT(**config_used)

    checkpoint_path = glob.glob("./{}/my_best_checkpoint*".format(path_exp))
    checkpoint = torch.load(checkpoint_path[0], map_location=torch.device(device))

    od = OrderedDict()
    for key in checkpoint["state_dict"].keys():
        # deleting prefix encoder from PL
        od[key.replace("atat.", "")] = checkpoint["state_dict"][key]

    model.load_state_dict(od)
    model.eval().to(device=device)

    # print('Use static features? {}'.format(config_used['general']['use_metadata']))
    # print('Use calculated features? {}'.format(config_used['use_features']))
    # print('Use QT? {}'.format(config_used['use_quantile_transformer']))

    # Generate batches over time
    data_dict_eval_time = dict()
    for eval_time in list_eval_time:
        # new_feat_col_total = get_tabular_data(feat_col, dict_add_feat_col, config_used, eval_time)
        if config_used["general"]["use_features"]:
            data_dict["extracted_feat"] = (
                torch.from_numpy(extracted_feat[eval_time]).float().unsqueeze(2)
            )

        data_dict_eval_time["test_{}".format(eval_time)] = obtain_valid_mask(
            data_dict.copy(),
            data_dict.copy()["mask"],
            data_dict.copy()["time"],
            eval_time,
        )

        data_dict_eval_time = get_chunks(
            data_dict_eval_time, eval_time, batch_size, config_used
        )

    pred_time, pred_prob_time = dict(), dict()
    for _, set_eval_time in enumerate(data_dict_eval_time.keys()):
        print(
            "#----------------- Testing in evaluation time: {} -----------------#".format(
                set_eval_time.upper()
            )
        )

        for i in range(len(data_dict_eval_time[set_eval_time]["data"])):
            print("Batch numero {}".format(i))

            batch = dict()
            batch["time"] = data_dict_eval_time[set_eval_time]["time"][i].to(device)
            batch["mask"] = data_dict_eval_time[set_eval_time]["mask"][i].to(device)

            if config_used["general"]["use_lightcurves"]:
                batch["data"] = data_dict_eval_time[set_eval_time]["data"][i].to(device)

            if config_used["general"]["use_lightcurves_err"]:
                batch["data_err"] = data_dict_eval_time[set_eval_time]["data_err"][
                    i
                ].to(device)

            tabular_features = []
            if config_used["general"]["use_metadata"]:
                tabular_features.append(
                    data_dict_eval_time[set_eval_time]["metadata_feat"][i].to(device)
                )

            if config_used["general"]["use_features"]:
                tabular_features.append(
                    data_dict_eval_time[set_eval_time]["extracted_feat"][i].to(device)
                )

            if tabular_features:
                batch["tabular_feat"] = torch.cat(tabular_features, axis=1)

            preds_lc, preds_tab, preds_mix = model(**batch)
            pred = (
                preds_mix
                if preds_mix is not None
                else (preds_lc if preds_lc is not None else preds_tab)
            )
            pred = torch.softmax(pred, dim=1)
            pred = pred.detach().cpu().numpy()

            preds_out = (
                np.concatenate([preds_out, np.argmax(pred, axis=1)], axis=0)
                if i != 0
                else np.argmax(pred, axis=1)
            )

            preds_prob_out = (
                np.concatenate([preds_prob_out, pred], axis=0) if i != 0 else pred
            )

        pred_time[set_eval_time] = preds_out
        pred_prob_time[set_eval_time] = preds_prob_out

    ################################################################################################

    # Save predictions
    print("Predictions saved in {}".format(path_exp))
    final_dict = {
        "SNID": SNID,
        "y_test": data_dict["labels"],
        "list_y_pred": pred_time.copy(),
        "list_y_pred_prob": pred_prob_time.copy(),
    }

    torch.save(final_dict, "./{}/all_predictions_times.pt".format(path_exp))

    del model


if __name__ == "__main__":
    data_name = sys.argv[1]

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    path_dict = {
        # La idea es que en algun momento se pueda utilizar para inferir cualquier dataset
        "elasticc": {
            "path_exp": "results/ELASTICC/BALTO",
            "data_root": "data/final/ELASTICC_2/LC_MD_FEAT",
            "list_eval_time": [8, 16, 32, 64, 128, 256, 512, 1024, 2048],
        },
        "ztf_ff": {
            "path_exp": "results/ZTF_ff/LC_MD_FEAT/ireyes_test_7/MTA",
            "data_root": "data/datasets/ZTF_ff/final/LC_MD_FEAT_240627_windows_200_12",
            "list_eval_time": [16, 32, 64, 128, 256, 512, 1024, 2048],
        },
    }

    path_exp = path_dict[data_name]["path_exp"]
    data_root = path_dict[data_name]["data_root"]
    list_eval_time = path_dict[data_name]["list_eval_time"]

    partition_used = 0
    batch_size = 256

    get_predictions(
        path_exp, data_root, list_eval_time, batch_size, device, partition_used
    )
