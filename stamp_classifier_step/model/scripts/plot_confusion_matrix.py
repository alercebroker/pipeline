import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from itertools import product


"""def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues, save_path=None):
"""

"""
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
"""

"""
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if not title:
        if normalize:
            title = 'Normalized confusion matrix Acc %.4f' % (np.trace(cm)/np.sum(cm))
        else:
            title = 'Confusion matrix, without normalization Acc %.4f' % (np.trace(cm)/np.sum(cm))


    # Only use the labels that appear in the data
    classes = classes[np.unique(np.stack([y_true, y_pred]))]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix Acc %.4f' % (np.trace(cm)/np.sum(cm)))
    else:
        print('Confusion matrix, without normalization Acc %.4f' % (np.trace(cm)/np.sum(cm)))

    print(cm)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    # fig.tight_layout()
    if save_path:
      fig.savefig(save_path)
    plt.show()
    return ax"""


class ConfusionMatrixDisplay(object):
    """Confusion Matrix visualization.
    It is recommend to use :func:`~sklearn.metrics.plot_confusion_matrix` to
    create a :class:`ConfusionMatrixDisplay`. All parameters are stored as
    attributes.
    Read more in the :ref:`User Guide <visualizations>`.
    Parameters
    ----------
    confusion_matrix : ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    display_labels : ndarray of shape (n_classes,)
        Display labels for plot.
    Attributes
    ----------
    im_ : matplotlib AxesImage
        Image representing the confusion matrix.
    text_ : ndarray of shape (n_classes, n_classes), dtype=matplotlib Text, \
            or None
        Array of matplotlib axes. `None` if `include_values` is false.
    ax_ : matplotlib Axes
        Axes with confusion matrix.
    figure_ : matplotlib Figure
        Figure containing the confusion matrix.
    """

    def __init__(self, confusion_matrix, display_labels):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    def plot(
        self,
        include_values=True,
        cmap="viridis",
        xticks_rotation="horizontal",
        values_format=None,
        ax=None,
        title=None,
        figsize=(8, 8),
        axis_fontsize=12,
        label_fontsize=12,
        colorbar=True,
    ):
        """Plot visualization.
        Parameters
        ----------
        include_values : bool, default=True
            Includes values in confusion matrix.
        cmap : str or matplotlib Colormap, default='viridis'
            Colormap recognized by matplotlib.
        xticks_rotation : {'vertical', 'horizontal'} or float, \
                         default='vertical'
            Rotation of xtick labels.
        values_format : str, default=None
            Format specification for values in confusion matrix. If `None`,
            the format specification is '.2f' for a normalized matrix, and
            'd' for a unnormalized matrix.
        ax : matplotlib axes, default=None
            Axes object to plot on. If `None`, a new figure and axes is
            created.
        Returns
        -------
        display : :class:`~sklearn.metrics.ConfusionMatrixDisplay`
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        self.text_ = None

        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = ".2g"

            # print text with appropriate color depending on background
            thresh = (cm.max() - cm.min()) / 2.0
            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                self.text_[i, j] = ax.text(
                    j,
                    i,
                    format(cm[i, j], values_format),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=label_fontsize,
                )

        if colorbar:
            fig.colorbar(self.im_, ax=ax)
        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=self.display_labels,
            yticklabels=self.display_labels,
        )
        ax.set_ylabel("True label", fontsize=label_fontsize)
        ax.set_xlabel("Predicted label", fontsize=label_fontsize)

        ax.tick_params(axis="both", labelsize=axis_fontsize)
        ax.set_title(title, fontsize=label_fontsize)

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self

    def savefig(self, save_path):
        plt.savefig(save_path, bbox_inches="tight")


class ConfusionMatrixSTD(ConfusionMatrixDisplay):
    def __init__(self, confusion_matrix, std_confusion_matrix, display_labels):
        super().__init__(confusion_matrix, display_labels)
        self.std_confusion_matrix = std_confusion_matrix

    def plot(
        self,
        include_values=True,
        cmap="viridis",
        xticks_rotation="horizontal",
        values_format=None,
        ax=None,
        title=None,
        figsize=(8, 8),
        axis_fontsize=12,
        label_fontsize=12,
        colorbar=True,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure

        cm = self.confusion_matrix
        cm_std = self.std_confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation="nearest", cmap=cmap)
        self.text_ = None

        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)

        if include_values:
            self.text_ = np.empty_like(cm, dtype=object)
            if values_format is None:
                values_format = ".2g"

            # print text with appropriate color depending on background
            thresh = (cm.max() - cm.min()) / 2.0
            for i, j in product(range(n_classes), range(n_classes)):
                color = cmap_max if cm[i, j] < thresh else cmap_min
                if cm[i, j] != 0:
                    self.text_[i, j] = ax.text(
                        j,
                        i,
                        format(cm[i, j], values_format)
                        + "$\pm$"
                        + str(np.round(cm_std[i, j], 2)),
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=label_fontsize,
                    )
                else:
                    self.text_[i, j] = ax.text(
                        j,
                        i,
                        format(cm[i, j], values_format),
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=label_fontsize,
                    )

        if colorbar:
            fig.colorbar(self.im_, ax=ax)
        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=self.display_labels,
            yticklabels=self.display_labels,
        )
        ax.set_ylabel("True label", fontsize=label_fontsize)
        ax.set_xlabel("Predicted label", fontsize=label_fontsize)

        ax.tick_params(axis="both", labelsize=axis_fontsize)
        ax.set_title(title, fontsize=label_fontsize)

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)

        self.figure_ = fig
        self.ax_ = ax
        return self.ax_


def plot_confusion_matrix(
    y_true,
    y_pred,
    classes=None,
    normalize=False,
    title=None,
    display_labels=None,
    include_values=True,
    xticks_rotation="horizontal",
    cmap=plt.cm.Blues,
    ax=None,
    figsize=(8, 8),
    axis_fontsize=12,
    label_fontsize=12,
    savepath=None,
    colorbar=True,
):
    plt.close("all")

    labels = classes

    cm = confusion_matrix(y_true, y_pred)
    if not title:
        if normalize:
            title = "Normalized confusion matrix Acc %.4f" % (np.trace(cm) / np.sum(cm))
        else:
            title = "Confusion matrix, without normalization Acc %.4f" % (
                np.trace(cm) / np.sum(cm)
            )

    # Only use the labels that appear in the data
    classes = labels[np.unique(np.stack([y_true, y_pred]))]
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix Acc %.4f" % (np.trace(cm) / np.sum(cm)))
    else:
        print(
            "Confusion matrix, without normalization Acc %.4f"
            % (np.trace(cm) / np.sum(cm))
        )

    # cm = np.round(cm, decimals=2)

    print(cm)

    display_labels = labels

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    pl = disp.plot(
        include_values=include_values,
        cmap=cmap,
        ax=ax,
        xticks_rotation=xticks_rotation,
        title=title,
        figsize=figsize,
        axis_fontsize=axis_fontsize,
        label_fontsize=label_fontsize,
        colorbar=colorbar,
    )

    if savepath is not None:
        disp.savefig(save_path=savepath)

    return pl


def plot_cm_std(
    mean_cm,
    std_cm,
    classes=None,
    normalize=False,
    title=None,
    display_labels=None,
    include_values=True,
    xticks_rotation="horizontal",
    cmap=plt.cm.Blues,
    ax=None,
    figsize=(8, 8),
    axis_fontsize=12,
    label_fontsize=12,
    savepath=None,
    colorbar=True,
):
    # plt.close("all")

    labels = classes
    cm = np.round(mean_cm, decimals=2)
    print(cm)

    display_labels = labels

    disp = ConfusionMatrixSTD(
        confusion_matrix=cm, display_labels=display_labels, std_confusion_matrix=std_cm
    )

    pl = disp.plot(
        include_values=include_values,
        cmap=cmap,
        ax=ax,
        xticks_rotation=xticks_rotation,
        title=title,
        figsize=figsize,
        axis_fontsize=axis_fontsize,
        label_fontsize=label_fontsize,
        colorbar=colorbar,
    )

    if savepath is not None:
        disp.savefig(save_path=savepath)

    return pl
