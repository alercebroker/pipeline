import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import pickle
import time

import copy

def preprocessing_features(metadata, oids_train, oids_val, oids_test):
    print('\nProcessing Features...')

    metadata = metadata.fillna(-999)
    metadata['non_detections'] = metadata['ncovhist'] - metadata['ndethist']
    clipping_rules = {
        'sgscore1': (-1, None),
        'distpsnr1': (-1, None),
        'sgscore2': (-1, None),
        'distpsnr2': (-1, None),
        'sgscore3': (-1, None),
        'distpsnr3': (-1, None),
        'fwhm': (None, 10),
        'ndethist': (None, 20),
        'ncovhist': (None, 3000),
        'chinr': (-1, 15),
        'sharpnr': (-1, 1.5),
        'non_detections': (None, 2000)
    }
    for feature, (min_val, max_val) in clipping_rules.items():
        if feature in metadata.columns:
            if min_val is not None:
                metadata[feature] = metadata[feature].clip(lower=min_val)
            if max_val is not None:
                metadata[feature] = metadata[feature].clip(upper=max_val)

    train_metadata = metadata.drop_duplicates(subset='oid', keep='first').set_index('oid').loc[oids_train]
    val_metadata = metadata.drop_duplicates(subset='oid', keep='first').set_index('oid').loc[oids_val]
    test_metadata = metadata.drop_duplicates(subset='oid', keep='first').set_index('oid').loc[oids_test]

    order_features = metadata.columns.difference(['oid', 'candid'])
    order_features = sorted(order_features)

    # 1. Calcular stats del set de entrenamiento
    norm_means = train_metadata[order_features].mean()
    norm_stds = train_metadata[order_features].std().replace(0, 1)

    # 2. Aplicar normalización a todos los conjuntos
    train_metadata[order_features] = (train_metadata[order_features] - norm_means) / norm_stds
    val_metadata[order_features]   = (val_metadata[order_features] - norm_means) / norm_stds
    test_metadata[order_features]  = (test_metadata[order_features]  - norm_means) / norm_stds

    md_processed = pd.concat([train_metadata, val_metadata, test_metadata])[order_features]
    
    return md_processed, order_features, norm_means, norm_stds


def get_tf_datasets(batch_size: int, use_metadata: bool, use_only_avro: bool):
    d = pd.read_pickle('./data/normalized_ndarrays.pkl')

    x_train = d['x_train']
    pos_train = d['pos_train']
    y_train = d['y_train']
    oids_train = d['oid_train']

    x_val = d['x_val']
    pos_val = d['pos_val']
    y_val = d['y_val']
    oids_val = d['oid_val']

    x_test = d['x_test']
    pos_test = d['pos_test']
    y_test = d['y_test']
    oids_test = d['oid_test']

    order_features = None
    norm_means, norm_stds = None, None
    if use_metadata:
        if use_only_avro:
            metadata = pd.read_parquet('data/full_stamp_classifier_metadata_hasavro.parquet')
        else:
            metadata_hasavro = pd.read_parquet('data/full_stamp_classifier_metadata_hasavro.parquet')
            metadata_noavro = pd.read_parquet('data/full_stamp_classifier_metadata_noavro.parquet')
            metadata = pd.concat([metadata_hasavro, metadata_noavro])

        metadata, order_features, norm_means, norm_stds = preprocessing_features(metadata, oids_train, oids_val, oids_test)
        print(f'Metadata columns: {list(metadata.columns)}')

        md_train = metadata.loc[oids_train]
        md_val = metadata.loc[oids_val]
        md_test = metadata.loc[oids_test]

    else:
        print(f'\nWe are using the coordenates:')
        md_train = copy.copy(pos_train)
        md_val = copy.copy(pos_val)
        md_test = copy.copy(pos_test)

    print(f'Metadata train: {md_train.shape}')
    print(f'Metadata val: {md_val.shape}')
    print(f'Metadata test: {md_test.shape}\n')    

    label_encoder = d['label_encoder']
    unique_labels = np.unique(y_train)
    dict_mapping_classes = {idx: label for idx, label in enumerate(label_encoder.classes_)}

    dict_info_model = {
        'dict_mapping_classes': dict_mapping_classes,
        'order_features': order_features,
        'norm_means': norm_means,
        'norm_stds': norm_stds, 
    }

    print('Stamps train:', x_train.shape)
    print('Stamps val:', x_val.shape)
    print('Stamps test:', x_test.shape)

    print('Stamps train:', y_train.shape)
    print('Stamps val:', y_val.shape)
    print('Stamps test:', y_test.shape)
    print('')

    n_classes = len(unique_labels)
    training_datasets_per_class = []
    for class_index in range(n_classes):
        class_slice = y_train == class_index
        class_dataset = (
            tf.data.Dataset.from_tensors(
                (x_train[class_slice], md_train[class_slice], y_train[class_slice]))
            .unbatch().repeat().shuffle(100, reshuffle_each_iteration=True).prefetch(20))
        training_datasets_per_class.append(class_dataset)
    training_dataset = tf.data.experimental.sample_from_datasets(
        training_datasets_per_class)
    training_dataset = training_dataset.batch(batch_size).prefetch(5)

    validation_dataset = tf.data.Dataset.from_tensors((x_val, md_val, y_val))
    test_dataset = tf.data.Dataset.from_tensors((x_test, md_test, y_test))

    validation_dataset = validation_dataset.unbatch().batch(batch_size).prefetch(5)
    test_dataset = test_dataset.unbatch().batch(batch_size).prefetch(5)
    return training_dataset, validation_dataset, test_dataset, dict_info_model