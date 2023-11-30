import pandas as pd
from lc_classifier.classifiers.ztf_mlp import ZTFClassifier


features = pd.read_parquet('data_231130/consolidated_features.parquet')
labels = pd.read_parquet('data_231130/labels_with_partitions.parquet')

list_of_classes = labels['astro_class'].unique()
list_of_classes.sort()

ztf_classifier = ZTFClassifier(list_of_classes)
config = {
    'learning_rate': 1e-4,
    'batch_size': 256
}
ztf_classifier.fit_from_features(features, labels, config)
ztf_classifier.save_classifier('ztf_classifier_model_231130')
