from matplotlib.patches import Polygon
from itertools import cycle
import torch.nn.functional as F
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    confusion_matrix,
    precision_score,
)
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import OneHotEncoder, label_binarize, LabelEncoder


def elasticc_confusion_matrix(
    y_true,
    y_pred,
    classes,
    normalize=False,
    title=None,
    plot_size=None,
    cmap=plt.cm.Blues,
    ax=None,
    all_range=True,
):
    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        normalize_factor = cm.sum(axis=1)[:, np.newaxis]
        cm = cm.astype("float") / normalize_factor
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
        cm = np.floor(cm).astype("int")

    if ax is not None:
        pass
    else:
        fig, ax = plt.subplots(figsize=plot_size, dpi=80)
    if normalize:
        im = ax.imshow(
            cm,
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1 if all_range else np.max(cm),
        )
    else:
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)

    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = (cm.max() - cm.min()) / 2.0 + cm.min()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    # fig.tight_layout()
    return ax  # , fig


def mean_confusion_matrix(
    y_true,
    y_pred,
    classes,
    normalize=False,
    title=None,
    plot_size=None,
    cmap=plt.cm.Blues,
    ax=None,
):
    """
    # y_true : [y_true_sim_1, y_true_sim_2, .... y_pred_sim_k]
    # y_pred : [y_pred_sim_1, y_pred_sim_2, .... y_pred_sim_k]
    # classes : ['label_1', label_2, ...] len(classes) = n_classes
    # title : -----
    # plot_size : (h, w)
    # cmap : plt.cm.Blues,  [
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    """

    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if not title:
        if normalize:
            title = "Normalized confusion matrix"
        else:
            title = "Confusion matrix, without normalization"

    # Compute confusion matrix

    n_sim = len(y_true)

    cnt = 0
    cm_list = []

    for y_true_idx, y_pred_idx in zip(y_true, y_pred):
        if cnt == 0:
            cm = confusion_matrix(y_true_idx, y_pred_idx)
        else:
            cm += confusion_matrix(y_true_idx, y_pred_idx)

        cm_list.append(confusion_matrix(y_true_idx, y_pred_idx))

        cnt += 1

    cm = np.mean(cm_list, axis=0)
    cm_std = np.std(cm_list, axis=0)

    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true_idx, y_pred_idx)]

    if normalize:
        normalize_factor = cm.sum(axis=1)[:, np.newaxis]

        cm = cm.astype("float") / normalize_factor
        cm_std = cm_std.astype("float") / normalize_factor

        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

        cm = np.floor(cm).astype("int")

    if ax is not None:
        pass
    else:
        fig, ax = plt.subplots(figsize=plot_size, dpi=80)

    if normalize:
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

    else:
        im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j:
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
            elif i == j:
                ax.text(
                    j,
                    i,
                    format(cm[i, j], fmt)
                    + "\n"
                    + r"$\pm$"
                    + format(cm_std[i, j], ".2f"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

    # fig.tight_layout()

    return ax
