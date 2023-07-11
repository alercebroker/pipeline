import pandas as pd
import pickle
import tqdm

import os
import sys

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_PATH)

df_with_metadata = pd.read_pickle(
    os.path.join(PROJECT_PATH, "..", "pickles", "bogus_and_features.pkl")
)
df_with_stamps = pd.read_pickle(
    os.path.join(PROJECT_PATH, "..", "pickles", "training_set_with_bogus.pkl")
)

for oid in tqdm(df_with_metadata.oid):
    df_with_metadata.at[
        df_with_metadata[df_with_metadata.oid == oid].index.values[0],
        "cutoutDifference",
    ] = df_with_stamps[df_with_stamps.oid == oid]["cutoutDifference"]
    df_with_metadata.at[
        df_with_metadata[df_with_metadata.oid == oid].index.values[0], "cutoutTemplate"
    ] = df_with_stamps[df_with_stamps.oid == oid]["cutoutTemplate"]
    df_with_metadata.at[
        df_with_metadata[df_with_metadata.oid == oid].index.values[0], "cutoutScience"
    ] = df_with_stamps[df_with_stamps.oid == oid]["cutoutScience"]

save_path = os.path.join(PROJECT_PATH, "..", "pickles", "bogus_and_featuresv2.pkl")
pickle.dump(df_with_metadata, open(save_path, "wb"), protocol=2)
