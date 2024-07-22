import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
from itertools import product
from alerce_classifiers.classifiers.mlp import MLPClassifier
from alerce_classifiers.classifiers.random_forest import RandomForestClassifier
from alerce_classifiers.classifiers.hierarchical_random_forest import (
    HierarchicalRandomForestClassifier,
)
from alerce_classifiers.classifiers.lightgbm import LightGBMClassifier
from alerce_classifiers.classifiers.xgboost import XGBoostClassifier

# from training import rename_feature


labels = pd.read_parquet(os.path.join("data_231206", "partitions.parquet"))
labels["aid"] = "aid_" + labels["oid"]
labels.set_index("aid", inplace=True)

list_of_classes = labels["alerceclass"].unique()
list_of_classes.sort()

labels_figure_order = [
    "SNIa",
    "SNIbc",
    "SNIIb",
    "SNII",
    "SNIIn",
    "SLSN",
    "TDE",
    "Microlensing",
    "QSO",
    "AGN",
    "Blazar",
    "YSO",
    "CV/Nova",
    "LPV",
    "EA",
    "EB/EW",
    "Periodic-Other",
    "RSCVn",
    "CEP",
    "RRLab",
    "RRLc",
    "DSCT",
]
assert set(list_of_classes) == set(labels_figure_order)

classifier_type = "HierarchicalRandomForest"
compute_predictions = True

if classifier_type == "MLP":
    classifier = MLPClassifier(list_of_classes)
    classifier.load_classifier("models/mlp_240416_wo_input_drop")
    predictions_filename = "mlp_predictions.parquet"
elif classifier_type == "RandomForest":
    classifier = RandomForestClassifier(list_of_classes)
    classifier.load_classifier("rf_classifier_240307")
elif classifier_type == "HierarchicalRandomForest":
    classifier = HierarchicalRandomForestClassifier(list_of_classes)
    model_dir = "models/hrf_classifier_20240722-162932"
    classifier.load_classifier(model_dir)
    predictions_filename = os.path.join(model_dir, "predictions.parquet")
elif classifier_type == "LightGBM":
    classifier = LightGBMClassifier(list_of_classes)
    classifier.load_classifier("lightgbm_classifier_240311")
elif classifier_type == "XGBoost":
    classifier = XGBoostClassifier(list_of_classes)
    classifier.load_classifier("xgboost_classifier_240312")
else:
    raise ValueError("invalid classifier type")


if compute_predictions:
    consolidated_features = pd.read_parquet(
        os.path.join("data_231206_ao_features", "consolidated_features.parquet")
    )

    shorten = consolidated_features["shorten"]
    consolidated_features = consolidated_features[
        [c for c in consolidated_features.columns if c != "shorten"]
    ]
    # consolidated_features.rename(columns=rename_feature, inplace=True)
    prediction_df = classifier.classify_batch(consolidated_features)
    prediction_df["shorten"] = shorten

    prediction_df.to_parquet(predictions_filename)

predictions = pd.read_parquet(predictions_filename)
logfile = os.path.join(model_dir, "classification_reports.txt")


def plot_cm_custom(ax, cm_mean, display_labels, title):
    n_classes = cm_mean.shape[0]
    im_kw = dict(interpolation="nearest", cmap=matplotlib.colormaps["Blues"])

    im_ = ax.imshow(cm_mean, **im_kw)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)
    text_ = np.empty_like(cm_mean, dtype=object)

    thresh = 0.5

    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm_mean[i, j] < thresh else cmap_min
        text_cm = f"{(100*cm_mean[i, j]):.1f}"
        text_[i, j] = ax.text(j, i, text_cm, ha="center", va="center", color=color)
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=display_labels,
        yticklabels=display_labels,
        ylabel="True label",
        xlabel="Predicted label",
    )

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation="vertical")
    ax.set_title(title)


def compute_stats(predictions_df, labels_df, ax, title):
    assert len(predictions_df) == len(labels_df)
    labels_df = labels_df.loc[predictions_df.index]
    true_astro_classes = labels_df["alerceclass"].values
    predicted_class = predictions_df.idxmax(axis=1).values
    with open(logfile, "a") as sys.stdout:
        print(title)
        print(classification_report(true_astro_classes, predicted_class))
    cm = confusion_matrix(
        true_astro_classes, predicted_class, labels=labels_figure_order
    )
    cm = cm / cm.sum(axis=1, keepdims=True)
    plot_cm_custom(ax, cm, labels_figure_order, title)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_astro_classes, predicted_class, average="macro"
    )
    return f1


all_shorten = predictions["shorten"].unique()
all_shorten = np.sort(all_shorten)

figure_dir = os.path.join(model_dir, "confusion_matrix")
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)

f1_dict = {}
for shorten in all_shorten:
    shorten_predictions = predictions[predictions["shorten"] == shorten]

    shorten_predictions = shorten_predictions[
        [c for c in shorten_predictions.columns if c != "shorten"]
    ]

    fig, ax = plt.subplots(1, 2, figsize=(15, 8), facecolor="white", dpi=100)
    plt.rcParams.update({"font.size": 8})

    val_labels = labels[labels["partition"] == "validation_0"]
    compute_stats(
        shorten_predictions.loc[val_labels.index],
        val_labels,
        ax[0],
        f"validation {shorten} days",
    )

    test_labels = labels[labels["partition"] == "test"]
    f1_test = compute_stats(
        shorten_predictions.loc[test_labels.index],
        test_labels,
        ax[1],
        f"test {shorten} days",
    )

    f1_dict[shorten] = f1_test
    plt.suptitle(str(shorten))
    plt.tight_layout()
    plt.savefig(os.path.join(figure_dir, f"{shorten}_days.png"))
    plt.close()

f1_dict["2000"] = f1_dict["None"]
del f1_dict["None"]

lc_lengths = [float(i) for i in f1_dict.keys()]
lc_f1 = f1_dict.values()

plt.scatter(lc_lengths, lc_f1)
plt.semilogx()
plt.xlabel("light curve max length [days]")
plt.ylabel("F1-score (macro)")
plt.savefig(os.path.join(figure_dir, f"f1_evolution.png"))
plt.close()
