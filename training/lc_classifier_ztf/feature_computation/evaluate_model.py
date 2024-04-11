import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from itertools import product
from lc_classifier.classifiers.ztf_mlp import ZTFClassifier
from lc_classifier.classifiers.random_forest import RandomForestClassifier
from lc_classifier.classifiers.lightgbm import LightGBMClassifier
from lc_classifier.classifiers.xgboost import XGBoostClassifier
from consolidate_features import get_shorten


dir_name = 'data_231206_ao_features'
labels = pd.read_parquet(os.path.join(dir_name, 'partitions.parquet'))
labels['aid'] = 'aid_' + labels['oid']
labels.set_index('aid', inplace=True)

list_of_classes = labels['alerceclass'].unique()
list_of_classes.sort()

labels_figure_order = [
    'SNIa',
    'SNIbc',
    'SNIIb',
    'SNII',
    'SNIIn',
    'SLSN',
    'TDE',
    'Microlensing',
    'QSO',
    'AGN',
    'Blazar',
    'YSO',
    'CV/Nova',
    'LPV',
    'EA',
    'EB/EW',
    'Periodic-Other',
    'RSCVn',
    'CEP',
    'RRLab',
    'RRLc',
    'DSCT'
]
assert set(list_of_classes) == set(labels_figure_order)

classifier_type = 'RandomForest'
output_filename = 'rf_predictions.parquet'

compute_predictions = False
if compute_predictions:
    if classifier_type == 'MLP':
        classifier = ZTFClassifier(list_of_classes)
        classifier.load_classifier('ztf_classifier_model_231206')
    elif classifier_type == 'RandomForest':
        classifier = RandomForestClassifier(list_of_classes)
        classifier.load_classifier('rf_classifier_240307')
    elif classifier_type == 'LightGBM':
        classifier = LightGBMClassifier(list_of_classes)
        classifier.load_classifier('lightgbm_classifier_240311')
    elif classifier_type == 'XGBoost':
        classifier = XGBoostClassifier(list_of_classes)
        classifier.load_classifier('xgboost_classifier_240312')
    else:
        raise ValueError('invalid classifier type')

    data_dir = os.listdir(dir_name)
    data_dir = [filename for filename in data_dir if 'astro_objects_batch' in filename]
    # data_dir = [filename for filename in data_dir if len(filename.split('_')) == 4]
    data_dir = sorted(data_dir)

    predictions = []
    for batch_filename in tqdm(data_dir):
        full_filename = os.path.join(dir_name, batch_filename)
        shorten = get_shorten(full_filename)
        astro_objects_batch = pd.read_pickle(full_filename)
        prediction_df = classifier.classify_batch(astro_objects_batch, return_dataframe=True)
        prediction_df['shorten'] = shorten
        predictions.append(prediction_df)

    predictions = pd.concat(predictions, axis=0)
    predictions.to_parquet(os.path.join(dir_name, output_filename))

predictions = pd.read_parquet(os.path.join(dir_name, output_filename))
print(predictions)


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
    true_astro_classes = labels_df['alerceclass'].values
    predicted_class = predictions_df.idxmax(axis=1).values
    print(title)
    print(classification_report(true_astro_classes, predicted_class))
    cm = confusion_matrix(true_astro_classes, predicted_class, labels=labels_figure_order)
    cm = cm / cm.sum(axis=1, keepdims=True)
    plot_cm_custom(ax, cm, labels_figure_order, title)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_astro_classes, predicted_class, average='macro')
    return f1


all_shorten = predictions['shorten'].unique()
all_shorten = np.sort(all_shorten)

f1_dict = {}
for shorten in all_shorten:
    shorten_predictions = predictions[predictions['shorten'] == shorten]

    shorten_predictions = shorten_predictions[[c for c in shorten_predictions.columns if c != 'shorten']]

    fig, ax = plt.subplots(1, 2, figsize=(15, 8), facecolor='white', dpi=100)
    plt.rcParams.update({'font.size': 8})

    val_labels = labels[labels['partition'] == 'validation_0']
    compute_stats(shorten_predictions.loc[val_labels.index], val_labels, ax[0], 'validation')

    test_labels = labels[labels['partition'] == 'test']
    f1_test = compute_stats(shorten_predictions.loc[test_labels.index], test_labels, ax[1], 'test')

    f1_dict[shorten] = f1_test
    plt.suptitle(str(shorten))
    plt.tight_layout()
    plt.show()

f1_dict['2000'] = f1_dict['None']
del f1_dict['None']

lc_lengths = [float(i) for i in f1_dict.keys()]
lc_f1 = f1_dict.values()

print(lc_f1)
print(lc_lengths)

plt.scatter(lc_lengths, lc_f1)
plt.semilogx()
plt.xlabel('light curve max length [days]')
plt.ylabel('F1-score (macro)')
plt.show()
