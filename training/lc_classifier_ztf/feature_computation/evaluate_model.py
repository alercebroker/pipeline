import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from itertools import product
from lc_classifier.classifiers.ztf_mlp import ZTFClassifier


labels = pd.read_parquet('data_231128/labels_with_partitions.parquet')
labels.set_index('aid', inplace=True)

list_of_classes = labels['astro_class'].unique()
list_of_classes.sort()

labels_figure_order = [
    'SNIa',
    'SNIbc',
    'SNII',
    'SNIIb',
    'SNIIn',
    'SLSN',
    'TDE',
    'Microlensing',

    #'nonSNIa',

    'QSO',
    'AGN',
    'Blazar',
    'YSO',
    'CVNova',
    'LPV',
    'EA',
    'EBEW',
    'Periodic-Other',
    'RSCVn',
    'CEP',
    'RRLab',
    'RRLc',
    'DSCT'
]
assert set(list_of_classes) == set(labels_figure_order)

ztf_classifier = ZTFClassifier(list_of_classes)
ztf_classifier.load_classifier('ztf_classifier_model_231128')

data_dir = os.listdir('data_231128')
data_dir = [filename for filename in data_dir if 'astro_objects_batch' in filename]
data_dir = sorted(data_dir)

predictions = []
for batch_filename in tqdm(data_dir):
    full_filename = os.path.join('data_231128', batch_filename)
    astro_objects_batch = pd.read_pickle(full_filename)
    prediction_df = ztf_classifier.classify_batch(astro_objects_batch, return_dataframe=True)
    predictions.append(prediction_df)

predictions = pd.concat(predictions, axis=0)


def plot_cm_custom(ax, cm_mean, display_labels, title):
    n_classes = cm_mean.shape[0]
    im_kw = dict(
        interpolation="nearest",
        cmap=plt.cm.get_cmap('Blues'))

    im_ = ax.imshow(cm_mean, **im_kw)
    cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)
    text_ = np.empty_like(cm_mean, dtype=object)

    thresh = 0.5

    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if cm_mean[i, j] < thresh else cmap_min
        text_cm = f'{(100*cm_mean[i, j]):.1f}'
        text_[i, j] = ax.text(
            j, i, text_cm, ha="center", va="center", color=color
        )
    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=display_labels,
        yticklabels=display_labels,
        ylabel="True label",
        xlabel="Predicted label",
    )

    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation='vertical')
    ax.set_title(title)


def compute_stats(predictions_df, labels_df, ax, title):
    assert len(predictions_df) == len(labels_df)
    labels_df = labels_df.loc[predictions_df.index]
    true_astro_classes = labels_df['astro_class'].values
    predicted_class = predictions_df.idxmax(axis=1).values
    print(classification_report(true_astro_classes, predicted_class))
    cm = confusion_matrix(true_astro_classes, predicted_class)
    cm = cm / cm.sum(axis=1, keepdims=True)
    plot_cm_custom(ax, cm, labels_figure_order, title)


fig, ax = plt.subplots(1, 2, figsize=(6, 6), facecolor='white')  # , dpi=100)
# plt.rcParams.update({'font.size': 12})

val_labels = labels[labels['partition'] == 'validation']
compute_stats(predictions.loc[val_labels.index], val_labels, ax[0], 'validation')

test_labels = labels[labels['partition'] == 'test']
compute_stats(predictions.loc[test_labels.index], test_labels, ax[1], 'test')

plt.tight_layout(pad=1.1)
plt.show()
