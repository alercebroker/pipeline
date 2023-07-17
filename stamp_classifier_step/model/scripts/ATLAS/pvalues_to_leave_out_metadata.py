import os
import sys

import numpy as np

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_PATH)


def get_feature_idxs_without_p(
    list_of_other_feature_names_to_leave_out=["STARDIST", "KASTDIST"]
):
    columns_path = os.path.join(PROJECT_PATH, "..", "ATLAS", "01", "COLUMNS")
    feature_name = []
    feature_list_idx_in_txt = []
    for line in open(columns_path):
        cols = line.split(" ")
        feature_list_idx_in_txt.append(int(cols[-1][:-1]))
        feature_name.append(cols[0])
    feature_list_idx_in_txt = np.array(feature_list_idx_in_txt) - 3
    # print(feature_name)
    # print(feature_list_idx_in_txt)
    feature_idxes_with_P = []
    for i in range(len(feature_name)):
        check_if_name_in_to_remove = np.sum(
            [
                name in feature_name[i]
                for name in list_of_other_feature_names_to_leave_out
            ]
        )
        if feature_name[i][0] == "P":
            feature_idxes_with_P.append(feature_list_idx_in_txt[i])
        elif check_if_name_in_to_remove != 0:
            feature_idxes_with_P.append(feature_list_idx_in_txt[i])
    # print(feature_idxes_with_P)
    feature_list_idxs = list(range(72))  # feature_list_idx_in_txt[2:]
    # print(feature_list_idxs)
    features_idxs_wihtout_P = [
        x for x in feature_list_idxs if x not in feature_idxes_with_P
    ]
    # print(features_idxs_wihtout_P)
    return features_idxs_wihtout_P


def get_feature_names():
    columns_path = os.path.join(PROJECT_PATH, "..", "ATLAS", "01", "COLUMNS")
    feature_name = []
    feature_list_idx_in_txt = []
    for line in open(columns_path):
        cols = line.split(" ")
        feature_list_idx_in_txt.append(int(cols[-1][:-1]))
        feature_name.append(cols[0])
    feature_name = feature_name[2:] + ["72"]
    # print(feature_name)
    return feature_name


if __name__ == "__main__":
    columns_path = os.path.join(PROJECT_PATH, "..", "ATLAS", "01", "COLUMNS")
    # columns_array = np.loadtxt(columns_path)
    # print(columns_array)
    feature_name = []
    feature_list_idx_in_txt = []
    for line in open(columns_path):
        cols = line.split(" ")
        feature_list_idx_in_txt.append(int(cols[-1][:-1]))
        feature_name.append(cols[0])
    feature_list_idx_in_txt = np.array(feature_list_idx_in_txt) - 3
    print(feature_name)
    print(feature_list_idx_in_txt)
    feature_idxes_with_P = []
    for i in range(len(feature_name)):
        if feature_name[i][0] == "P":
            feature_idxes_with_P.append(feature_list_idx_in_txt[i])
    print(feature_idxes_with_P)
    feature_list_idxs = list(range(73))  # feature_list_idx_in_txt[2:]
    print(feature_list_idxs)
    features_to_use = [x for x in feature_list_idxs if x not in feature_idxes_with_P]
    print(features_to_use)
    # metadata_IDs.append(cols[0])
