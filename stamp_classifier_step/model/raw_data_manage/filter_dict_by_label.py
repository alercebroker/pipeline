import os
import sys
import pickle as pkl
import pandas as pd
import gc

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)


def load_pickle(path):
    infile = open(path, "rb")
    data = pkl.load(infile)
    return data


def save_pickle(dict, path):
    with open(path, "wb") as handle:
        pkl.dump(dict, handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    data_dict_path = os.path.join(
        PROJECT_PATH, "..", "..", "finalstamps", "final_stamps.pickle"
    )
    save_filtered_path = os.path.join(
        PROJECT_PATH,
        "..",
        # 'baseline_classifier', 'training_set_v1',
        "datasets",
        "filtered_stamps_by_label.pkl",
    )
    labels_path = os.path.join(
        PROJECT_PATH,
        "..",
        # 'baseline_classifier', 'training_set_v1',
        "datasets",
        "labels.pkl",
    )

    labels_dict = pd.read_pickle(labels_path)
    data_dict = pd.read_pickle(data_dict_path)
    print(labels_dict.head())
    print(list(data_dict.keys())[:10])

    filtered_data_dict = {}
    label_oids = ["tar"] + list(labels_dict["classALeRCE"].keys())
    data_oids = list(data_dict.keys())

    n_labels_without_stamps = 0
    for single_label_oid in label_oids:
        if single_label_oid in data_oids:
            filtered_data_dict[single_label_oid] = data_dict[single_label_oid]
        else:
            n_labels_without_stamps += 1

    print("labels wihtout stamps %i\n" % n_labels_without_stamps)
    print("total samples %i\n" % int(len(list(filtered_data_dict.keys())) - 1))
    #
    # data_dict.clear()
    # gc.collect()

    save_pickle(filtered_data_dict, save_filtered_path)
