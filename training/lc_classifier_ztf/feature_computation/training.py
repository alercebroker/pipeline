import pandas as pd
from lc_classifier.classifiers.ztf_mlp import ZTFClassifier


features = pd.read_parquet('data_231206_features/consolidated_features.parquet')
labels = pd.read_parquet('data_231206_features/partitions.parquet')

# support for shortened lightcurves
features.index = (features.index.values + '_' + features['shorten'].astype(str)).values

label_suffixes = features['shorten'].astype(str).unique()
features = features[[c for c in features.columns if c != 'shorten']]

label_list = []
for suffix in label_suffixes:
    labels_copy = labels.copy()
    # labels_copy['aid'] += '_' + suffix
    labels_copy['aid'] = 'aid_' + labels_copy['oid'] + '_' + suffix
    label_list.append(labels_copy)

labels = pd.concat(label_list, axis=0)

labels.rename(columns={'alerceclass': 'astro_class'}, inplace=True)
list_of_classes = labels['astro_class'].unique()
list_of_classes.sort()

ztf_classifier = ZTFClassifier(list_of_classes)
config = {
    'learning_rate': 1e-4,
    'batch_size': 512
}
ztf_classifier.fit_from_features(features, labels, config)
ztf_classifier.save_classifier('ztf_classifier_model_231206')
