import os
import sys
import pickle as pkl
import pandas as pd
import numpy as np

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname("__file__"), ".."))
sys.path.append(PROJECT_PATH)


def save_pickle(dict, path):
    with open(path, "wb") as handle:
        pkl.dump(dict, handle, protocol=pkl.HIGHEST_PROTOCOL)


def get_SNE_label_dict(label_df):
    SNE_label_dict = {}
    SNE_count = 0
    label_names = np.unique(labels_df["classALeRCE"].values)
    SN_labels = [s for s in label_names if "SN" in s]
    oids = list(labels_df["classALeRCE"].keys())
    labels_dict = labels_df["classALeRCE"]
    for single_oid in oids:
        object_label = labels_dict[single_oid]
        if object_label in SN_labels:
            SNE_count += 1
            # print({'label': 1, 'classALeRCE': object_label})
            SNE_label_dict[single_oid] = {"label": 1, "classALeRCE": object_label}
        elif object_label not in SN_labels:
            SNE_label_dict[single_oid] = {"label": 0, "classALeRCE": object_label}
        else:
            raise ValueError("oid_name label dont match SNe or non SNe")
    print("SNe label count %i" % int(SNE_count))
    return SNE_label_dict


def pair_images_and_labels(stamps_dict, labels_dict):
    paired_dict = {"images": [], "labels": [], "classALeRCE": [], "oid": []}
    oids = list(stamps_dict.keys())
    for single_oid in oids:
        paired_dict["images"].append(stamps_dict[single_oid]["stamps"])
        paired_dict["labels"].append(labels_dict[single_oid]["label"])
        paired_dict["classALeRCE"].append(labels_dict[single_oid]["classALeRCE"])
        paired_dict["oid"].append(single_oid)
    # paired_dict['labels'] = np.array(paired_dict['labels'])
    # paired_dict['classALeRCE'] = np.array(paired_dict['classALeRCE'])
    # paired_dict['oid'] = np.array(paired_dict['oid'])
    return paired_dict


if __name__ == "__main__":
    data_path = os.path.join(
        PROJECT_PATH,
        "..",
        "datasets",
        "ZTF",
        "stamp_classifier",
        "first_epoch_stamps.pkl",
    )
    labels_path = os.path.join(
        PROJECT_PATH, "..", "datasets", "ZTF", "stamp_classifier", "labels.pkl"
    )
    save_paired_data_path = os.path.join(
        PROJECT_PATH, "..", "datasets", "ZTF", "stamp_classifier", "ztf_dataset.pkl"
    )
    stamps_dict = pd.read_pickle(data_path)
    labels_df = pd.read_pickle(labels_path)
    labels_dict = get_SNE_label_dict(labels_df)
    dataset_dict = pair_images_and_labels(stamps_dict, labels_dict)
    save_pickle(dataset_dict, save_paired_data_path)
