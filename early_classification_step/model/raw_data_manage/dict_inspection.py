import os
import sys
import pickle as pkl
import pandas as pd

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)


def load_pickle(path):
    infile = open(path, "rb")
    data = pkl.load(infile)
    return data


if __name__ == "__main__":
    data_path = os.path.join(
        PROJECT_PATH,
        "..",
        "..",
        "..",
        "..",
        "storage",
        "ztf_workspace",
        "finalstamps",
        "final_stamps.pickle",
    )
    print(os.path.abspath(data_path))
    data_dict = pd.read_pickle(data_path)
    print(data_dict.keys())
