# data_loader.py
import os
import glob
import mlflow
import logging
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder

# Local imports
from src.data.stamp_processing import (
    prepare_model_input, 
    crop_stamps_ndarray, 
    normalize_batches, 
    get_max_hw, 
    apply_min_max_scaling
)
from src.data.metadata_processing import process_coordinates, apply_normalization


# --- Helper Processing Functions ---

def compute_global_stats(data, args):
    """Calculates mean and std for channel-wise normalization using training data."""
    logging.info("Calculating global stats for channel-wise normalization...")
    
    stamps = data[args['stamps_cols']].values
    max_h, max_w = get_max_hw(stamps)
    
    padded_stamps, _ = prepare_model_input(stamps, max_h, max_w)
    
    if args['cropping']['use']:
        padded_stamps = crop_stamps_ndarray(padded_stamps, args['cropping']['crop_size'])
        
    padded_stamps = np.nan_to_num(padded_stamps, nan=0.0)
    scaled_stamps = apply_min_max_scaling(padded_stamps)
    
    mean_global = np.mean(scaled_stamps, axis=(0, 1, 2))
    std_global = np.std(scaled_stamps, axis=(0, 1, 2))
    
    stats = {'mean': mean_global.reshape(1, 1, 1, -1), 'std': std_global.reshape(1, 1, 1, -1)}
    logging.info(f"Global Stats Calculated - Mean: {mean_global.flatten()}, Std: {std_global.flatten()}")
    return stats


def flip_negative_flux(padded_stamps, flux):
    """Flips the stamps with negative flux values to positive (usually for difference channel)."""
    # Assuming channel 1 is 'difference'
    negative_flux_mask = flux < 0
    if np.any(negative_flux_mask):
        logging.info(f"Flipping {np.sum(negative_flux_mask)} stamps with negative flux values to positive.")
        padded_stamps[negative_flux_mask, :, :, 1] *= -1
    else:
        logging.info("No stamps with negative flux values found.")
    return padded_stamps


def check_stamp_shapes(stamps):
    """Utility to debug stamp shapes."""
    for i, row in enumerate(stamps):
        for j, stamp in enumerate(row):
            if hasattr(stamp, "shape"):
                if stamp.ndim != 2:
                    logging.warning(f"⚠️ Stamp at row {i}, col {j} has strange shape {stamp.shape}")
            else:
                logging.error(f"❌ Stamp at row {i}, col {j} has no attribute 'shape', type: {type(stamp)}")


def process_stamp(data, args, global_stats=None):
    """Prepares and normalizes image data (stamps)."""
    stamps = data[args['stamps_cols']].values
    max_h, max_w = get_max_hw(stamps)
    
    padded_stamps, padding_masks = prepare_model_input(stamps, max_h, max_w)
    
    if args['cropping']['use']:
        padded_stamps = crop_stamps_ndarray(padded_stamps, args['cropping']['crop_size'])
        padding_masks = crop_stamps_ndarray(padding_masks, args['cropping']['crop_size'])

    # Handle negative flux / isdiffpos
    if 'isdiffpos' in data.columns:
        negative_flux_mask = data['isdiffpos'] == -1
        if np.any(negative_flux_mask):
            logging.info(f"Flipping {np.sum(negative_flux_mask)} stamps with negative flux values to positive (based on isdiffpos).")
            padded_stamps[negative_flux_mask, :, :, 1] *= -1
        else:
            logging.info("No stamps with negative flux values found.")
    else:
        if 'psfFlux' in data.columns:
            # logging.debug(f"Using psfFlux for flipping check.")
            padded_stamps = flip_negative_flux(padded_stamps, data['psfFlux'].values)
        else:
            raise ValueError("No difference column (isdiffpos or psfFlux) found in data.")

    norm_type = args.get('stamp_norm_type', 'legacy')
    logging.info(f"Using stamp normalization type: '{norm_type}'")

    padded_stamps = normalize_batches(
        padded_stamps, 
        padding_masks, 
        args['batch_size'],
        norm_type=norm_type,
        global_stats=global_stats
    )
    return padded_stamps


def process_metadata(
        data, 
        args, 
        dict_info_model, 
        norm_type='z-score', 
        path_norm_dir='./normalization_params', 
        is_test_only=False
    ):
    """
    Procesa metadata asegurando que siempre existan todas las bandas definidas,
    independientemente de si aparecen o no en el dataset actual.
    """
    # 1. Leer configuración
    use_coords = args.get('use_coords', False)
    use_band = args.get('use_band', False)
    band_col = args.get('band_col', 'band')
    
    # Lista maestra de bandas (Default a Rubin si no está en config)
    known_bands = args.get('known_bands', ['u', 'g', 'r', 'i', 'z', 'y'])

    # 2. Selección inicial de columnas
    cols_to_keep_temp = [args['id_col']]
    if use_band: cols_to_keep_temp.append(band_col)
    if use_coords: 
        cols_to_keep_temp.append(args['ra_col'])
        cols_to_keep_temp.append(args['dec_col'])

    metadata = data[cols_to_keep_temp].copy()
    metadata = metadata.set_index(args['id_col'])

    metadata_final = pd.DataFrame(index=metadata.index)

    # 3. Procesar Coordenadas
    if use_coords:
        coord_df = process_coordinates(
            oids=metadata.index,
            ra=metadata[args['ra_col']].values,
            dec=metadata[args['dec_col']].values,
            coord_type=args['coord_type'],
        )
        metadata_final = pd.concat([metadata_final, coord_df], axis=1)

    # 4. Procesar Banda (LÓGICA CORREGIDA)
    if use_band:
        if band_col in metadata.columns:
            # AQUI ESTA EL TRUCO:
            # Convertimos la columna a Categorical forzando las categorías conocidas.
            # Si una banda no está en los datos, pandas igual sabe que existe.
            metadata[band_col] = pd.Categorical(
                metadata[band_col], 
                categories=known_bands
            )
            
            # get_dummies ahora generará 6 columnas siempre (band_u, band_g, ... band_y)
            # Las bandas que no aparezcan en 'data' serán columnas llenas de ceros.
            dummies = pd.get_dummies(metadata[band_col], prefix='band', dtype='float32')
            
            # Guardamos el orden para referencia, aunque ahora es fijo por 'known_bands'
            if not is_test_only:
                dict_info_model['band_columns'] = dummies.columns.tolist()
                logging.info(f"Bandas fijadas por configuración: {known_bands}")
            
            metadata_final = pd.concat([metadata_final, dummies], axis=1)
        else:
            # Si se pide banda pero no hay columna, creamos las 6 columnas en cero
            # para no romper la arquitectura del modelo
            logging.warning(f"⚠️ Columna '{band_col}' no encontrada. Rellenando con ceros.")
            for b in known_bands:
                metadata_final[f'band_{b}'] = 0.0

    # --- Legacy / Optionals (kept commented) ---
    # metadata = metadata.drop(columns=[args['ra_col'], args['dec_col']])
    # metadata = metadata.fillna(-999)
    # metadata = fill_and_clipping_metadata(metadata) # ZTF specific
    
    # 5. Ordenar características para consistencia
    if not is_test_only:
        # Si no hay metadata seleccionada (ambos False), esto estará vacío
        ordered_cols = sorted(metadata_final.columns)
        dict_info_model['order_features'] = ordered_cols
    else:
        ordered_cols = dict_info_model['order_features']

    metadata_final = metadata_final[ordered_cols]

    # Feature Normalization (kept commented as in original)
    # metadata = apply_normalization(
    #    metadata,
    #    norm_type=norm_type,
    #    dict_info_model=dict_info_model,
    #    path_norm_dir=path_norm_dir,
    #    is_test_only=is_test_only
    # )

    if not is_test_only:
        return metadata_final, dict_info_model
    else:
        return metadata_final


# --- Main Data Loading Function ---

def get_tf_datasets(batch_size: int, args: dict, load_pretrained_model: bool = False):
    args_loader = args['loader']
    dict_info_model = dict()  

    logging.info("--- Starting Data Loading Process ---")

    # 1. Load Partition
    partition = pd.read_parquet(f"{args['path_partition']}")
    logging.info(f"Classes found in partition: {partition['class'].unique()}")

    # 2. Load Stamp Data chunks
    data = []
    path_chunks = glob.glob(f"{args['dir_data']}/stamps/*.pkl")
    for path_chunk in path_chunks:
        data.append(pd.read_pickle(path_chunk))
    data = pd.concat(data)

    # 3. Load Objects (Metadata)
    objs = pd.read_parquet(
        f"{args['dir_data']}/comm_objs_4candmax.parquet"
    )

    objs = objs[[
        args_loader['id_col'], 
        args_loader['candid_col'], 
        'class',
        'psfFlux',
        'ra',
        'dec',
        args_loader['band_col'], 
    ]]
    objs['psfFlux'] = objs['psfFlux'].astype(float)

    # 4. Merge Data
    data = pd.merge(
        data,
        objs,
        left_on=[args_loader['id_col'], args_loader['candid_col']],
        right_on=[args_loader['id_col'], args_loader['candid_col']],
        how='left'
    )

    if data[args_loader['band_col']].dtype != 'object' and data[args_loader['band_col']].dtype.name != 'string':
        data[args_loader['band_col']] = data[args_loader['band_col']].astype(int)
        data[args_loader['band_col']] = data[args_loader['band_col']].replace(
            {
                1: 'g', 
                2: 'r', 
                3: 'i', 
                4: 'z', 
                5: 'y', 
                6: 'u'
            })

    if 'sn_vs_others' in args['path_partition']:
        logging.info("Applying binary mapping (SN vs Others)...")
        data['class'] = data['class'].replace({
            'AGN': 'Others',
            'VS': 'Others',
            'asteroid': 'Others',
            'bogus': 'Others',
        })

    # 5. Split Data based on Partition (Fold)
    fold = args['loader']['fold']
    logging.info(f"Processing Fold: {fold}")

    # -- Train Split --
    oids_train = partition[partition['partition'] == f'training_{fold}'][args_loader['id_col']].tolist()

    # -- Validation Split (Real + Synth) --
    df_val_total = partition[partition['partition'] == f'validation_{fold}']
    df_val_real = df_val_total[df_val_total['dataset_origin'] == 'real']
    df_val_synth = df_val_total[df_val_total['dataset_origin'] == 'synthetic']

    first_obs_val_real_df = pd.DataFrame()
    if not df_val_real.empty:
        idx_first_obs_val = df_val_real.dropna(subset=['mjd']).groupby(args_loader['id_col'])['mjd'].idxmin()
        first_obs_val_real_df = df_val_real.loc[idx_first_obs_val]

    first_obs_val_df = pd.concat([first_obs_val_real_df, df_val_synth], ignore_index=True)
    oids_val = first_obs_val_df[args_loader['id_col']].tolist()
    candid_val = first_obs_val_df[args_loader['candid_col']].tolist()

    # -- Test Split --
    df_test_real = partition[
        (partition['partition'] == 'test') & 
        (partition['dataset_origin'] == 'real')
    ]
    idx_first_obs_test = df_test_real.loc[df_test_real.dropna(subset=['mjd']).groupby(args_loader['id_col'])['mjd'].idxmin()].index
    first_obs_test_df = df_test_real.loc[idx_first_obs_test]
    oids_test = first_obs_test_df[args_loader['id_col']].tolist()
    candid_test = first_obs_test_df[args_loader['candid_col']].tolist()
    
    # Filter DataFrames
    data_train = data[data[args_loader['id_col']].isin(oids_train)]
    data_val = data[(data[args_loader['id_col']].isin(oids_val)) & 
                     (data[args_loader['candid_col']].isin(candid_val))]
    data_test = data[(data[args_loader['id_col']].isin(oids_test)) & 
                     (data[args_loader['candid_col']].isin(candid_test))]

    # Save sample for deployment checks
    data_name = "data_sample_test.pkl"
    data_test.head(10).to_pickle(data_name)
    mlflow.log_artifact(data_name, artifact_path="deployment_checks")

    #data_test.head(10).to_pickle(f"{args['artifact_path']}/deployment_checks/data_sample_test.pkl")

    # 6. Handle Simulated Data
    simulated_data = args.get('simulated_data', {})
    
    if simulated_data.get('use', False):
        logging.info("Using Simulated Data (Config mode)...")
        synth_partition = partition[partition['dataset_origin'] == 'synthetic']  

        #print(synth_partition)  
        
        # Verificar si la columna 'synth_file_path' existe en el DataFrame
        if 'synth_file_path' in synth_partition.columns and not synth_partition['synth_file_path'].isna().all():
            # Usar synth_file_path si existe y tiene valores
            train_files_ids = synth_partition[synth_partition[args_loader['id_col']].isin(oids_train)]['synth_file_path'].dropna().tolist()
            val_files_ids = synth_partition[synth_partition[args_loader['id_col']].isin(oids_val)]['synth_file_path'].dropna().tolist()
        else:
            # Usar oid si no existe synth_file_path
            train_files_ids = synth_partition[synth_partition[args_loader['id_col']].isin(oids_train)][args_loader['id_col']].tolist()
            val_files_ids = synth_partition[synth_partition[args_loader['id_col']].isin(oids_val)][args_loader['id_col']].tolist()

        synth_path = simulated_data['path']
        synthetic_total_SN = pd.read_pickle(f'{synth_path}')
        synthetic_total_SN['class'] = 'SN' 
        synthetic_total_SN['psfFlux'] = 1 
        # Temporal placeholders
        synthetic_total_SN['ra'] = 0 
        synthetic_total_SN['dec'] = 0 


        # Filtrar datos de entrenamiento y validación sintéticos
        if 'synth_file_path' in synthetic_total_SN.columns and not synthetic_total_SN['synth_file_path'].isna().all():
            # Usar synth_file_path para filtrar
            data_train_synth = synthetic_total_SN[synthetic_total_SN['synth_file_path'].isin(train_files_ids)]
            data_val_synth = synthetic_total_SN[synthetic_total_SN['synth_file_path'].isin(val_files_ids)]
        else:
            # Usar oid para filtrar
            data_train_synth = synthetic_total_SN[synthetic_total_SN[args_loader['id_col']].isin(train_files_ids)]
            data_val_synth = synthetic_total_SN[synthetic_total_SN[args_loader['id_col']].isin(val_files_ids)]

        # Combinar con datos reales
        data_train = pd.concat([data_train, data_train_synth], ignore_index=True)
        data_val = pd.concat([data_val, data_val_synth], ignore_index=True)
        
        # Preparar datos de prueba sintéticos
        # Primero obtener los oids de test sintéticos
        oids_test_sim = partition[
            (partition['partition'] == 'test') &
            (partition['dataset_origin'] == 'synthetic')
        ][args_loader['id_col']].tolist()
        
        # Obtener los IDs de archivos para test
        if 'synth_file_path' in synth_partition.columns and not synth_partition['synth_file_path'].isna().all():
            test_files_ids = synth_partition[synth_partition[args_loader['id_col']].isin(oids_test_sim)]['synth_file_path'].dropna().tolist()
        else:
            test_files_ids = synth_partition[synth_partition[args_loader['id_col']].isin(oids_test_sim)][args_loader['id_col']].tolist()
        
        # Filtrar datos de test sintéticos
        if 'synth_file_path' in synthetic_total_SN.columns and not synthetic_total_SN['synth_file_path'].isna().all():
            data_test_sim = synthetic_total_SN[synthetic_total_SN['synth_file_path'].isin(test_files_ids)]
        else:
            data_test_sim = synthetic_total_SN[synthetic_total_SN[args_loader['id_col']].isin(test_files_ids)]
        
        logging.info(f"Synthetic Data Added - Train: {len(data_train_synth)}, Val: {len(data_val_synth)}, Test: {len(data_test_sim)}")
        logging.info(f"Total Train after adding synthetic: {len(data_train)}")

    else:
        logging.info("Using Legacy/Local Simulated Data path...")
        # Hardcoded legacy path (kept as requested)
        partition_aux = pd.read_parquet('/home/dmoreno/pipeline_v4_final/pipeline/training/stamp_classifier/data_acquisition/rubin/data/processed/partitions/partitions_trainSN_valSN_synthetic_eval_firstStamp_v1/partitions.parquet')
        synth_partition = partition_aux[partition_aux['dataset_origin'] == 'synthetic']     
        synthetic_total_SN = pd.read_pickle(f'/home/dmoreno/pipeline_v4_final/pipeline/training/stamp_classifier/data_acquisition/rubin/data/simulated_data/synthetic_total_SN.pkl')

        print('Hay que arreglarlo')
        exit()

        synthetic_total_SN['class'] = 'SN' 
        synthetic_total_SN['psfFlux'] = 1 
        synthetic_total_SN['ra'] = 0 
        synthetic_total_SN['dec'] = 0 

        oids_test_sim = partition_aux[
            (partition_aux['partition'] == 'test') &
            (partition_aux['dataset_origin'] == 'synthetic')
        ][args_loader['id_col']].tolist()

        test_files_ids = synth_partition[synth_partition[args_loader['id_col']].isin(oids_test_sim)]['synth_file_path'].tolist()
        data_test_sim = synthetic_total_SN[synthetic_total_SN['synth_file_path'].isin(test_files_ids)]

    # 7. Sync OIDs and Split Features/Labels
    logging.info("Syncing OID lists with final data order...")
    oids_train = data_train[args_loader['id_col']].tolist()
    oids_val = data_val[args_loader['id_col']].tolist()
    oids_test = data_test[args_loader['id_col']].tolist()

    candid_train = data_train[args_loader['candid_col']].tolist()
    candid_val = data_val[args_loader['candid_col']].tolist()
    candid_test = data_test[args_loader['candid_col']].tolist()

    ra_train = data_train['ra'].tolist()
    ra_val = data_val['ra'].tolist()
    ra_test = data_test['ra'].tolist()
    ra_test_sim = data_test_sim['ra'].tolist()

    dec_train = data_train['dec'].tolist()
    dec_val = data_val['dec'].tolist()
    dec_test = data_test['dec'].tolist()
    dec_test_sim = data_test_sim['dec'].tolist()

    class_col = args_loader['class_col']
    def split_data_labels(df):
        return df.drop(columns=[class_col]), df[class_col].values

    X_train, y_train = split_data_labels(data_train)
    X_val, y_val = split_data_labels(data_val)
    X_test, y_test = split_data_labels(data_test)
    X_test_sim, y_test_sim = split_data_labels(data_test_sim) # Always exists based on logic above

    # 8. Process Stamps
    stamp_norm_type = args_loader.get('stamp_norm_type', 'legacy')
    global_stats = None

    if stamp_norm_type == 'channel':
        global_stats = compute_global_stats(X_train, args_loader)

    stamp_train = process_stamp(X_train, args_loader, global_stats=global_stats)
    stamp_val = process_stamp(X_val, args_loader, global_stats=global_stats)
    stamp_test = process_stamp(X_test, args_loader, global_stats=global_stats)
    stamp_test_sim = process_stamp(X_test_sim, args_loader, global_stats=global_stats)

    # 9. Process Metadata
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
    md_test_sim = process_metadata(
        X_test_sim, args_loader, dict_info_model,
        norm_type=norm_type, path_norm_dir=args['artifact_path'], is_test_only=True
    )

    # Convert to float32 numpy
    md_train = md_train.to_numpy(dtype='float32')
    md_val = md_val.to_numpy(dtype='float32')
    md_test = md_test.to_numpy(dtype='float32')
    md_test_sim = md_test_sim.to_numpy(dtype='float32')

    # --- Debugging Block (Matching Indices) ---
    # Kept as requested
    # tol = 1e-3
    # match_matrix = np.all(np.isclose(md_train.values[:, None, :], md_test.values[None, :, :], atol=tol), axis=2)
    # hay_match = np.any(match_matrix)
    # train_idx, test_idx = np.where(match_matrix)
    # for ti, te in zip(train_idx, test_idx):
    #    print(f"Train idx {ti}  <->  Test idx {te}")
    # ... (rest of debugging prints)

    # 10. Encode Labels
    if not load_pretrained_model:
        logging.info(f"Unique classes - Train: {np.unique(y_train)}, Val: {np.unique(y_val)}, Test: {np.unique(y_test)}")
        
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train)
        y_val = label_encoder.transform(y_val)
        y_test = label_encoder.transform(y_test)
        y_test_sim = label_encoder.transform(y_test_sim)

        dict_mapping_classes = {
            idx: class_label for idx, class_label in enumerate(label_encoder.classes_)
        }
    else:
        logging.info("Using existing class mapping from pretrained model.")
        dict_mapping_classes = args['dict_mapping_classes']
        class_to_idx = {class_label: idx for idx, class_label in dict_mapping_classes.items()}

        try:
            y_train = np.array([class_to_idx[label] for label in y_train])
            y_val = np.array([class_to_idx[label] for label in y_val])
            y_test = np.array([class_to_idx[label] for label in y_test])
            y_test_sim = np.array([class_to_idx[label] for label in y_test_sim])
        except KeyError as e:
            logging.error(f"Error: Label '{e.args[0]}' found in data but not in pretrained class mapping.")
            raise

    dict_info_model.update({'dict_mapping_classes': dict_mapping_classes})
    logging.info(f"Class Mapping: {dict_mapping_classes}")

    # 11. Final Summary
    logging.info("🧾 Dataset summary:")
    logging.info(f"""
    ┌────────────────────────────────────────────┐
    │ Stamps train   : {str(stamp_train.shape):<25} │
    │ Stamps val     : {str(stamp_val.shape):<25} │
    │ Stamps test    : {str(stamp_test.shape):<25} │
    │ Labels train   : {str(y_train.shape):<25} │
    │ Labels val     : {str(y_val.shape):<25} │
    │ Labels test    : {str(y_test.shape):<25} │
    │ Labels test sim: {str(y_test_sim.shape):<25} │
    │ Metadata train : {str(md_train.shape):<25} │
    │ Metadata val   : {str(md_val.shape):<25} │
    │ Metadata test  : {str(md_test.shape):<25} │
    └────────────────────────────────────────────┘
    """)


    # 12. Create TF Datasets
    # -- Balanced Training Dataset --
    n_classes = len(np.unique(y_train))
    training_datasets_per_class = []
    
    for class_index in range(n_classes):
        class_slice = y_train == class_index
        inputs = (stamp_train[class_slice], md_train[class_slice])
        labels = y_train[class_slice]

        class_dataset = (
            tf.data.Dataset.from_tensors((inputs, labels))
            .unbatch()
            .repeat()
            .shuffle(100, reshuffle_each_iteration=True)
            .prefetch(20)
        )
        training_datasets_per_class.append(class_dataset)
        
    training_dataset = tf.data.experimental.sample_from_datasets(training_datasets_per_class)
    training_dataset = training_dataset.batch(batch_size).prefetch(5)

    # -- Evaluation Datasets --
    def create_eval_ds(stamps, metas, labels):
        ds = tf.data.Dataset.from_tensors(((stamps, metas), labels))
        return ds.unbatch().batch(batch_size).prefetch(5)

    train_ds_for_eval = create_eval_ds(stamp_train, md_train, y_train)
    validation_dataset = create_eval_ds(stamp_val, md_val, y_val)
    test_dataset = create_eval_ds(stamp_test, md_test, y_test)
    test_dataset_sim = create_eval_ds(stamp_test_sim, md_test_sim, y_test_sim)
    
    # 13. Pack results into a Dictionary
    data_pack = {
        'datasets': {
            'train': training_dataset,
            'train_eval': train_ds_for_eval,
            'val': validation_dataset,
            'test': test_dataset,
            'test_sim': test_dataset_sim
        },
        'identifiers': {
            'oids': {
                'train': oids_train,
                'val': oids_val,
                'test': oids_test,
                'test_sim': oids_test_sim
            },
            'candid': {
                'train': candid_train,
                'val': candid_val,
                'test': candid_test
            },

            'ra': {
                'train': ra_train,
                'val': ra_val,
                'test': ra_test,
                'test_sim': ra_test_sim
            },
            'dec': {
                'train': dec_train,
                'val': dec_val,
                'test': dec_test,
                'test_sim': dec_test_sim
            },

            'files': {
                'test_sim': test_files_ids
            }
        },
        'info': dict_info_model
    }

    return data_pack