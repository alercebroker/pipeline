import os
import pandas as pd
import time
from lc_classifier.classifiers.mlp import MLPClassifier
from lc_classifier.classifiers.random_forest import RandomForestClassifier
from lc_classifier.classifiers.hierarchical_random_forest import HierarchicalRandomForestClassifier
from lc_classifier.classifiers.lightgbm import LightGBMClassifier
from lc_classifier.classifiers.xgboost import XGBoostClassifier


# def rename_feature(feature_name: str):
#     split_feature_name = feature_name.split('_')
#     fid_map = {"g": "_1", "r": "_2", "g,r": "_12", 'nan': ""}
#     band_suffix = fid_map[split_feature_name[-1]]
#     feature_name = '_'.join(split_feature_name[:-1]) + band_suffix
#     feature_name = feature_name.replace('-', '_')
#     feature_name = feature_name.replace('/', '_')
#     return feature_name


if __name__ == '__main__':

    features = pd.read_parquet('data_231206_ao_features/consolidated_features.parquet')
    labels = pd.read_parquet('data_231206_ao_features/partitions.parquet')

    # support for shortened lightcurves
    features.index = (features.index.values + '_' + features['shorten'].astype(str)).values

    label_suffixes = features['shorten'].astype(str).unique()
    features = features[[c for c in features.columns if c != 'shorten']]

    # rename features to match avro schema
    # features.rename(columns=rename_feature, inplace=True)

    label_list = []
    for suffix in label_suffixes:
        labels_copy = labels.copy()
        # labels_copy['aid'] += '_' + suffix
        labels_copy['aid'] = 'aid_' + labels_copy['oid'] + '_' + suffix
        label_list.append(labels_copy)

    labels = pd.concat(label_list, axis=0)

    labels.rename(columns={'alerceclass': 'astro_class'}, inplace=True)
    labels.set_index('aid', inplace=True)
    list_of_classes = labels['astro_class'].unique()
    list_of_classes.sort()

    classifier_type = 'HierarchicalRandomForest'
    models_dir = "models"
    timestr = time.strftime("%Y%m%d-%H%M%S")

    if classifier_type == 'MLP':
        ztf_classifier = MLPClassifier(list_of_classes)
        config = {
            'learning_rate': 1e-4,
            'batch_size': 4096
        }
        ztf_classifier.fit(features, labels, config)
        ztf_classifier.save_classifier(
            os.path.join(models_dir, f'mlp_{timestr}'))
    elif classifier_type == 'RandomForest':
        classifier = RandomForestClassifier(list_of_classes)
        config = {
            'n_trees': 500,
            'n_jobs': 8,
            'verbose': 11
        }
        classifier.fit_from_features(features, labels, config)

        test_labels = labels[labels['partition'] == 'test']
        test_features = features.loc[test_labels['aid'].values]

        test_probs = classifier.classify_batch_from_features(test_features)
        test_labels.set_index('aid', inplace=True)
        a = pd.concat([test_labels, test_probs.idxmax(axis=1)], axis=1)
        acc = a.groupby('astro_class').apply(
            lambda x: (x['astro_class'] == x[0]).astype(float).mean()
        )
        print(acc)

        classifier.save_classifier('rf_classifier_240307')
    elif classifier_type == 'HierarchicalRandomForest':
        classifier = HierarchicalRandomForestClassifier(list_of_classes)

        sampling_strategy = {
            'Top': dict(zip(
                ['transient', 'periodic', 'stochastic'],
                [1_000*len(label_suffixes)]*3)),
            'Transient': 'auto',
            'Periodic': dict(zip(
                classifier.class_hierarchy['periodic'],
                [200*len(label_suffixes)]*len(classifier.class_hierarchy['periodic']))),
            'Stochastic': 'auto'
        }

        config = {
            'n_trees': 500,
            'max_depth': 10,
            'n_jobs': 8,
            'verbose': 11,
            'sampling': sampling_strategy
        }
        classifier.fit(features, labels, config)

        test_labels = labels[labels['partition'] == 'test']
        test_features = features.loc[test_labels.index]

        test_probs = classifier.classify_batch(test_features)
        a = pd.concat([test_labels, test_probs.idxmax(axis=1)], axis=1)
        acc = a.groupby('astro_class').apply(
            lambda x: (x['astro_class'] == x[0]).astype(float).mean()
        )
        print(acc)
        classifier.save_classifier(
            os.path.join(models_dir, f'hrf_classifier_{timestr}'))
    elif classifier_type == 'LightGBM':
        classifier = LightGBMClassifier(list_of_classes)
        config = {}
        classifier.fit_from_features(features, labels, config)

        test_labels = labels[labels['partition'] == 'test']
        test_features = features.loc[test_labels['aid'].values]

        test_probs = classifier.classify_batch_from_features(test_features)
        test_labels.set_index('aid', inplace=True)
        a = pd.concat([test_labels, test_probs.idxmax(axis=1)], axis=1)
        acc = a.groupby('astro_class').apply(
            lambda x: (x['astro_class'] == x[0]).astype(float).mean()
        )
        print(acc)
        classifier.save_classifier('lightgbm_classifier_240311')
    elif classifier_type == 'XGBoost':
        classifier = XGBoostClassifier(list_of_classes)
        config = {}
        classifier.fit_from_features(features, labels, config)

        test_labels = labels[labels['partition'] == 'test']
        test_features = features.loc[test_labels['aid'].values]

        test_probs = classifier.classify_batch_from_features(test_features)
        test_labels.set_index('aid', inplace=True)
        a = pd.concat([test_labels, test_probs.idxmax(axis=1)], axis=1)
        acc = a.groupby('astro_class').apply(
            lambda x: (x['astro_class'] == x[0]).astype(float).mean()
        )
        print(acc)
        classifier.save_classifier('xgboost_classifier_240312')
