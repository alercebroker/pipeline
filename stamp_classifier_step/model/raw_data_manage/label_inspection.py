import os
import sys
import pickle as pkl
import pandas as pd
import numpy as np

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)


def load_pickle(path):
    infile = open(path, "rb")
    data = pkl.load(infile)
    return data


if __name__ == "__main__":
    # dict_path = os.path.join(PROJECT_PATH, '..', '..', '..', '..', 'storage',
    #                          'ztf_workspace', 'finalstamps',
    #                          'final_stamps.pickle')
    labels_path = os.path.join(
        PROJECT_PATH,
        "..",
        # 'baseline_classifier', 'training_set_v1',
        "datasets",
        "ZTF",
        "stamp_classifier",
        "labels.pkl",
    )
    # print(os.path.abspath(labels_path))
    labels_dict = pd.read_pickle(labels_path)
    # print(list(labels_dict.keys()))
    print(labels_dict.head())
    oids = list(labels_dict["classALeRCE"].keys())
    print(oids)
    if "ZTF18aaaarlg" in oids:
        print("is")
    # labels_dict['cl']
    label_names = np.unique(labels_dict["classALeRCE"].values)
    # ['AGN', 'CV', 'Ceph', 'DSCT', 'EBC', 'EBSD/D', 'LPV', 'Novae',
    # 'Periodic-Other', 'RRL', 'SLSN', 'SNeII', 'SNeIIb', 'SNeIIn',
    # 'SNeIa', 'SNeIa-sub', 'SNeIb/c', 'TDE']
