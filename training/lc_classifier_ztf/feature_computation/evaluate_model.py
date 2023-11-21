import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from lc_classifier.classifiers.ztf_mlp import ZTFClassifier


labels = pd.read_parquet('data/labels_with_partitions.parquet')
labels['aid'] = 'aid_' + labels['aid']
labels.set_index('aid', inplace=True)

list_of_classes = labels['astro_class'].unique()
list_of_classes.sort()

ztf_classifier = ZTFClassifier(list_of_classes)
ztf_classifier.load_classifier('ztf_classifier_model')

data_dir = os.listdir('data')
data_dir = [filename for filename in data_dir if 'astro_objects_batch' in filename]
data_dir = sorted(data_dir)

predictions = []
for batch_filename in tqdm(data_dir):
    full_filename = os.path.join('data', batch_filename)
    astro_objects_batch = pd.read_pickle(full_filename)
    prediction_df = ztf_classifier.classify_batch(astro_objects_batch, return_dataframe=True)
    predictions.append(prediction_df)

predictions = pd.concat(predictions, axis=0)


def compute_stats(predictions_df, labels_df):
    assert len(predictions_df) == len(labels_df)
    labels_df = labels_df.loc[predictions_df.index]
    true_astro_classes = labels_df['astro_class'].values
    predicted_class = predictions_df.idxmax(axis=1).values
    print(classification_report(true_astro_classes, predicted_class))
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_astro_classes, predicted_class, average='macro')
    print(precision, recall, f1)


val_labels = labels[labels['partition'] == 'validation']
compute_stats(predictions.loc[val_labels.index], val_labels)

test_labels = labels[labels['partition'] == 'test']
compute_stats(predictions.loc[test_labels.index], test_labels)
