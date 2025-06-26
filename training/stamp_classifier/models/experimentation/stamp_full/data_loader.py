import numpy as np
import pandas as pd
import tensorflow as tf

import copy

from sklearn.preprocessing import QuantileTransformer

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


def process_coordinates(metadata, pos, coord_type):
    """
    Dispatch function to process celestial coordinates.
    """
    if coord_type == "cartesian":    
        drop_cols = ['ra', 'dec', 'gal_l', 'gal_b', 'ecl_long', 'ecl_lat'] 
        metadata = metadata.drop(columns=drop_cols, errors='ignore') 
        df_coord = pd.DataFrame(pos, index=metadata.index, columns=['pos_x', 'pos_y', 'pos_z'])
        metadata = pd.concat([metadata, df_coord], axis=1)
        return metadata

    elif coord_type == "spherical":
        return metadata
    
    else:
        raise ValueError(f"Unknown coord_type: {coord_type}. Use 'cartesian', 'spherical', or 'none'.")


def fill_and_clipping_metadata(metadata):
    metadata = metadata.fillna(-999)
    
    cols_name = metadata.columns.difference(['oid', 'candid'])
    for feature, (min_val, max_val) in clipping_rules.items():
        if feature in cols_name:
            mask_missing = metadata[feature] == -999
            if min_val is not None:
                metadata.loc[~mask_missing, feature] = metadata.loc[~mask_missing, feature].clip(lower=min_val)
            if max_val is not None:
                metadata.loc[~mask_missing, feature] = metadata.loc[~mask_missing, feature].clip(upper=max_val)
            metadata.loc[mask_missing, feature] = -999
    return metadata


def split_metadata(metadata, oids, candid):
    index = list(zip(oids, candid))
    df_split = metadata.set_index(['oid', 'candid']).loc[index]
    df_split = df_split.reset_index().drop_duplicates(subset='oid', keep='first').set_index('oid')
    return df_split


def qt_normalization(df, qt):
    mask_valid = ~(df == -999).any(axis=1)
    df_valid = df.loc[mask_valid].copy()
    df_missing = df.loc[~mask_valid].copy()

    df_valid = pd.DataFrame(qt.transform(df_valid) + 0.1, index=df_valid.index, columns=df.columns)

    df_missing = df_missing.replace(-999, 0.0)
    df_out = pd.concat([df_valid, df_missing])
    df_out = df_out.reindex(df.index)
    return df_out


def zscore_normalization(df, norm_means, norm_stds):
    mask_valid = ~(df == -999).any(axis=1)
    df_valid = df.loc[mask_valid].copy()
    df_missing = df.loc[~mask_valid].copy()

    df_valid = (df_valid - norm_means) / norm_stds

    df_missing = df_missing.replace(-999, -10)
    df_out = pd.concat([df_valid, df_missing])
    df_out = df_out.reindex(df.index)
    return df_out


def normalize_metadata(metadata_train, metadata_val, metadata_test, norm_type, dict_info_model):
    mask_valid = ~(metadata_train == -999).any(axis=1)
    metadata_train_valid = metadata_train.loc[mask_valid].copy()

    qt, norm_means, norm_stds = None, None, None
    if norm_type == 'QT':
        qt = QuantileTransformer(output_distribution='uniform', random_state=42, subsample=5000)
        qt.fit(metadata_train_valid)

        metadata_train = qt_normalization(metadata_train, qt)
        metadata_val = qt_normalization(metadata_val, qt)
        metadata_test = qt_normalization(metadata_test, qt)
        
    elif norm_type == 'z-score':
        norm_means = metadata_train_valid.mean()
        norm_stds = metadata_train_valid.std().replace(0, 1)

        metadata_train = zscore_normalization(metadata_train, norm_means, norm_stds)
        metadata_val = zscore_normalization(metadata_val, norm_means, norm_stds)
        metadata_test = zscore_normalization(metadata_test, norm_means, norm_stds)

    dict_info_model.update({
        'qt': qt,
        'norm_means': norm_means if norm_means is None else [float(v) for v in norm_means.values],
        'norm_stds': norm_stds if norm_stds is None else [float(v) for v in norm_stds.values],
    })



    return metadata_train, metadata_val, metadata_test


def preprocess_features(metadata, data, args_general):
    metadata['non_detections'] = metadata['ncovhist'] - metadata['ndethist']
    metadata = fill_and_clipping_metadata(metadata)

    metadata_train = split_metadata(metadata, oids=data['oid_train'], candid=data['candid_train'])
    metadata_val = split_metadata(metadata, oids=data['oid_val'], candid=data['candid_val'])
    metadata_test = split_metadata(metadata, oids=data['oid_test'], candid=data['candid_test'])

    metadata_train = process_coordinates(metadata_train, data['pos_train'], args_general['coord_type'])
    metadata_val = process_coordinates(metadata_val, data['pos_val'], args_general['coord_type'])
    metadata_test = process_coordinates(metadata_test, data['pos_test'], args_general['coord_type'])

    # Determine and enforce column order once #######
    order_features = metadata_train.columns.difference(['oid', 'candid'])
    order_features = sorted(order_features)
    metadata_train = metadata_train[order_features]
    metadata_val   = metadata_val[order_features]
    metadata_test  = metadata_test[order_features]

    dict_info_model = {
        'order_features': order_features
    }
    metadata_train, metadata_val, metadata_test = normalize_metadata(
                                                                    metadata_train, 
                                                                    metadata_val, 
                                                                    metadata_test, 
                                                                    args_general['norm_type'],
                                                                    dict_info_model
                                                                    )
    
    return metadata_train, metadata_val, metadata_test, dict_info_model



def get_tf_datasets(batch_size: int, args_general: dict):
    data = pd.read_pickle(args_general['path_data'])
    print(f'data.keys(): {data.keys()}')

    x_train = data['x_train']
    y_train = data['y_train']

    x_val = data['x_val']
    y_val = data['y_val']

    x_test = data['x_test']
    y_test = data['y_test']

    if args_general['use_metadata']:
        if args_general['path_data'].find('hasavro') != -1:
            metadata = pd.read_parquet('data/full_stamp_classifier_metadata_hasavro.parquet')
        else:
            metadata_hasavro = pd.read_parquet('data/full_stamp_classifier_metadata_hasavro.parquet')
            metadata_noavro = pd.read_parquet('data/full_stamp_classifier_metadata_noavro.parquet')
            metadata = pd.concat([metadata_hasavro, metadata_noavro])

        if args_general['add_new_sats_sn']:
            new_sats_and_sn = pd.read_parquet('data/new_sats_sne_250610/new_sats_sne_250610_metadata.parquet')
            metadata = pd.concat([metadata, new_sats_and_sn]).reset_index(drop=True)

        md_train, md_val, md_test, dict_info_model = preprocess_features(metadata, data, args_general)
        md_train = md_train.loc[data['oid_train']]
        md_val = md_val.loc[data['oid_val']]
        md_test = md_test.loc[data['oid_test']]

        print(f'Metadata columns: {list(metadata.columns)}')

    else:
        print(f'\nWe are using the coordenates:')
        dict_info_model = {
            'order_features': ['pos_x', 'pos_y', 'pos_z'],
            'qt': None,
            'norm_means': None,
            'norm_stds': None,
        }
        md_train = copy.copy(data['pos_train'])
        md_val = copy.copy(data['pos_val'])
        md_test = copy.copy(data['pos_test'])


    label_encoder = data['label_encoder']
    unique_labels = np.unique(y_train)
    dict_mapping_classes = {idx: label for idx, label in enumerate(label_encoder.classes_)}

    dict_info_model.update({
        'dict_mapping_classes': dict_mapping_classes,
    })


    print('Stamps train:', x_train.shape)
    print('Stamps val:', x_val.shape)
    print('Stamps test:', x_test.shape)

    print('Stamps train:', y_train.shape)
    print('Stamps val:', y_val.shape)
    print('Stamps test:', y_test.shape)
    print('')

    print(f'Metadata train: {md_train.shape}')
    print(f'Metadata val: {md_val.shape}')
    print(f'Metadata test: {md_test.shape}\n')  

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