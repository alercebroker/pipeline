import pandas as pd
from sklearn.model_selection import train_test_split


features = pd.read_parquet("data_231130/consolidated_features.parquet")
labels = pd.read_parquet("data_231130/objects_231130.parquet")
labels.reset_index(inplace=True)
labels.rename(columns={"alerceclass": "astro_class", "oid": "aid"}, inplace=True)
labels = labels[["aid", "astro_class"]]
labels["aid"] = "aid_" + labels["aid"]
labels.set_index("aid", inplace=True)
labels = labels.loc[features.index]

# make partitions
seed = 0
aid_train_val, aid_test = train_test_split(
    labels.index.values,
    stratify=labels["astro_class"].values,
    test_size=0.2,
    random_state=seed,
)

labels_train_val = labels.loc[aid_train_val]
aid_train, aid_val = train_test_split(
    aid_train_val,
    stratify=labels_train_val["astro_class"].values,
    train_size=0.75,
    random_state=seed,
)

labels_training = labels.loc[aid_train].copy()
labels_training["partition"] = "training"

labels_validation = labels.loc[aid_val].copy()
labels_validation["partition"] = "validation"

labels_test = labels.loc[aid_test].copy()
labels_test["partition"] = "test"

labels = pd.concat([labels_training, labels_validation, labels_test])
labels.index.name = "aid"
labels.reset_index(inplace=True)

labels.to_parquet("data_231130/labels_with_partitions.parquet")
