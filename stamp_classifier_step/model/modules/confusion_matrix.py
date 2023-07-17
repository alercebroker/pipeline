import matplotlib
import matplotlib.pyplot as plt
import numpy as np


from sklearn.metrics import confusion_matrix

class_names_conf_mat = np.array(["AGN", "SN", "VS", "asteroid", "bogus"])
class_names_real_bog_conf_mat = np.array(["bogus", "real"])
class_names_atlas_conf_mat = np.array(
    ["cr", "streak", "burn", "scar", "kast", "spike", "noise"]
)
class_names_atlas_short_conf_mat = np.array(["artifact", "kast", "streak"])


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    show=False,
    normalize=False,
    title=None,
    cmap=plt.cm.Blues,
    save_path=None,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if show == False:
        matplotlib.use("agg")
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if not title:
        if normalize:
            title = "Normalized confusion matrix Acc %.4f" % (np.trace(cm) / np.sum(cm))
        else:
            title = "Confusion matrix, without normalization Acc %.4f" % (
                np.trace(cm) / np.sum(cm)
            )

    n_labels = len(np.unique(np.stack([y_true, y_pred])))
    if class_names is None and n_labels == len(class_names_conf_mat):
        class_names = class_names_conf_mat
    elif class_names is None and n_labels == len(class_names_real_bog_conf_mat):
        class_names = class_names_real_bog_conf_mat
    elif class_names is None and n_labels == len(class_names_atlas_conf_mat):
        class_names = class_names_atlas_conf_mat
    elif class_names is None and n_labels == len(class_names_atlas_short_conf_mat):
        class_names = class_names_atlas_short_conf_mat
    else:
        class_names = np.unique(np.stack([y_true, y_pred]))

    # Only use the labels that appear in the data
    class_names = class_names[np.unique(np.stack([y_true, y_pred])).astype("int")]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix Acc %.4f" % (np.trace(cm) / np.sum(cm)))
    else:
        print(
            "Confusion matrix, without normalization Acc %.4f"
            % (np.trace(cm) / np.sum(cm))
        )

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=class_names,
        yticklabels=class_names,
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
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()
    return ax
