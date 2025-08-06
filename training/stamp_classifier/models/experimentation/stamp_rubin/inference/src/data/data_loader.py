import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import glob

from sklearn.preprocessing import LabelEncoder

from src.data.stamp_processing import prepare_model_input, crop_stamps_ndarray, normalize_batches, get_max_hw
from src.data.metadata_processing import process_coordinates, apply_normalization

import logging
#logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def check_stamp_shapes(stamps):
    for i, row in enumerate(stamps):  # fila por fila
        for j, stamp in enumerate(row):  # columna por columna (science, ref, diff)
            if hasattr(stamp, "shape"):
                if stamp.ndim != 2:
                    print(f"⚠️ Estampilla en fila {i}, columna {j} tiene shape {stamp.shape}")
            else:
                print(f"❌ Estampilla en fila {i}, columna {j} no tiene atributo 'shape', tipo: {type(stamp)}")


def process_stamp(data, args):
    stamps = data[args['stamps_cols']].values
    max_h, max_w = get_max_hw(stamps)
    padded_stamps, padding_masks = prepare_model_input(stamps, max_h, max_w)
    if args['cropping']['use']:
        padded_stamps = crop_stamps_ndarray(padded_stamps, args['cropping']['crop_size'])
        padding_masks = crop_stamps_ndarray(padding_masks, args['cropping']['crop_size'])
    padded_stamps = normalize_batches(padded_stamps, padding_masks, args['batch_size'])
    return padded_stamps


def process_metadata(
        data, 
        args, 
        dict_info_model, 
        norm_type='z-score', 
        path_norm_dir='./normalization_params', 
        is_test_only=False
        ):
    
    columns_to_rm = args['stamps_cols'] + [args['candid_col']] + ["target_name"] # nuevo
    metadata = data.drop(columns=columns_to_rm, axis=1)
    metadata = metadata.set_index(args['id_col'])

    coord_df = process_coordinates(
        oids=metadata.index,
        ra=metadata[args['ra_col']].values,
        dec=metadata[args['dec_col']].values,
        coord_type=args['coord_type'],
    )
    #metadata = pd.concat([metadata, coord_df], axis=1)
    metadata = metadata.drop(columns=[args['ra_col'], args['dec_col']])
    metadata = metadata.fillna(-999)
    
    # Esto es solo si se quiere usar con ZTF
    #metadata = fill_and_clipping_metadata(metadata)
    
    if not is_test_only:
        ordered_cols = sorted(metadata.columns)
        dict_info_model['order_features'] = ordered_cols
    else:
        ordered_cols = dict_info_model['order_features']
    metadata = metadata[ordered_cols]

    metadata = apply_normalization(
        metadata,
        norm_type=norm_type,
        dict_info_model=dict_info_model,
        path_norm_dir=path_norm_dir,
        is_test_only=is_test_only
    )

    # CREO QUE EL DEVOLVER DICT INFO MODEL ESTA INNECESARIO
    if not is_test_only:
        return metadata, dict_info_model
    else:
        return metadata

def get_tf_datasets(batch_size: int, args: dict):
    # Dictionary config saved
    args_loader = args['loader']
    dict_info_model = dict()  

    # Load partition
    partition = pd.read_parquet(f"{args['path_partition']}")

    # Load data
    # Usar MULTIPROCESSING SI HAY MUCHOS CHUNKS
    data = []
    path_chunks = glob.glob(f"{args['dir_data']}/stamps/*.pkl")
    for path_chunk in path_chunks:
        data.append(pd.read_pickle(path_chunk))
    data = pd.concat(data)
    #data = pd.read_pickle(f"{args['dir_data']}/firststamps_250708.pkl")
    print(data.columns)

    # Split data
    fold = args['loader']['fold']
    oids_train = partition[partition['partition'] == f'training_{fold}'][args_loader['id_col']].tolist()
    oids_val = partition[partition['partition'] == f'validation_{fold}'][args_loader['id_col']].tolist()
    oids_test = partition[partition['partition'] == 'test'][args_loader['id_col']].tolist()

    data_train = data[data[args_loader['id_col']].isin(oids_train)]
    data_val = data[data[args_loader['id_col']].isin(oids_val)]
    data_test = data[data[args_loader['id_col']].isin(oids_test)]

    # --- Separate data and labels ---
    class_col = args_loader['class_col']
    def split_data_labels(df):
        return df.drop(columns=[class_col]), df[class_col].values
    
    X_train, y_train = split_data_labels(data_train)
    X_val, y_val = split_data_labels(data_val)
    X_test, y_test = split_data_labels(data_test)

    del data_train, data_val, data_test

    # Stamps input
    stamp_train = process_stamp(X_train, args_loader)
    stamp_val = process_stamp(X_val, args_loader)
    stamp_test = process_stamp(X_test, args_loader)

    # Metadata input
    norm_type = args_loader.get('norm_type', 'z-score')
    md_train, dict_info_model = process_metadata(
        X_train, args_loader, dict_info_model,
        norm_type=norm_type, path_norm_dir=args['artifact_path'], is_test_only=False
    )

    md_val = process_metadata(
        X_val, args_loader, dict_info_model,
        norm_type=norm_type, path_norm_dir=args['artifact_path'], is_test_only=True
    )

    md_test = process_metadata(
        X_test, args_loader, dict_info_model,
        norm_type=norm_type, path_norm_dir=args['artifact_path'], is_test_only=True
    )

    # --- Encode labels ---
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    dict_mapping_classes = {
        idx: class_label for idx, class_label in enumerate(label_encoder.classes_)
    }
    dict_info_model.update({
        'label_encoder': label_encoder,
        'dict_mapping_classes': dict_mapping_classes,
    })

    # Reemplazos de prints actuales:
    logging.info("🧾 Dataset summary:")
    logging.info(
        f"""
    ┌────────────────────────────────────────────┐
    │ Stamps train   : {str(stamp_train.shape):<25} │
    │ Stamps val     : {str(stamp_val.shape):<25} │
    │ Stamps test    : {str(stamp_test.shape):<25} │
    │ Labels train   : {str(y_train.shape):<25} │
    │ Labels val     : {str(y_val.shape):<25} │
    │ Labels test    : {str(y_test.shape):<25} │
    │ Metadata train : {str(md_train.shape):<25} │
    │ Metadata val   : {str(md_val.shape):<25} │
    │ Metadata test  : {str(md_test.shape):<25} │
    └────────────────────────────────────────────┘
    """
    )

    n_classes = len(np.unique(y_train))
    training_datasets_per_class = []
    for class_index in range(n_classes):
        class_slice = y_train == class_index
        inputs = (stamp_train[class_slice], md_train[class_slice])
        labels = y_train[class_slice]

        class_dataset = (
            tf.data.Dataset.from_tensors((inputs, labels))
            .unbatch()
            #.shuffle(10000, reshuffle_each_iteration=False)
            .repeat()
            .shuffle(100, reshuffle_each_iteration=True)
            .prefetch(20)
        )
        
        training_datasets_per_class.append(class_dataset)
    training_dataset = tf.data.experimental.sample_from_datasets(
        training_datasets_per_class)
    training_dataset = training_dataset.batch(batch_size).prefetch(5)

    validation_dataset = tf.data.Dataset.from_tensors(((stamp_val, md_val), y_val))
    test_dataset = tf.data.Dataset.from_tensors(((stamp_test, md_test), y_test))

    validation_dataset = validation_dataset.unbatch().batch(batch_size).prefetch(5)
    test_dataset = test_dataset.unbatch().batch(batch_size).prefetch(5)
    
    return training_dataset, validation_dataset, test_dataset, dict_info_model