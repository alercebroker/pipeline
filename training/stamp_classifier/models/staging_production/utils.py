import numpy as np
import pandas as pd
import logging
import glob
import tqdm

from multiprocessing import Pool, cpu_count

from lc_classifier.features.core.base import query_ao_table

def get_InputDTO(ao_lightcurve, ao_feature):
    detections = get_det_AO_to_InputDTO(ao_lightcurve)
    features = get_feat_AO_to_InputDTO(ao_feature)
    return detections, features

def recover_mag_and_emag_with_signed_flux(df):
    df['isdiffpos'] = np.sign(df['fluxdiff_uJy']).astype(int)
    df['mag'] = -2.5 * np.log10(np.abs(df['fluxdiff_uJy'])) + 23.9
    df['e_mag'] = df['fluxerrdiff_uJy'] / np.abs(df['fluxdiff_uJy'])
    df.loc[df['fluxdiff_uJy'] == 0, ['mag', 'e_mag']] = np.nan
    return df

def get_det_AO_to_InputDTO(lightcurve):
    detections = lightcurve['detections']
    detections = detections[detections.unit == 'diff_flux'].rename(columns={
        'brightness': 'fluxdiff_uJy',
        'e_brightness': 'fluxerrdiff_uJy',
    })

    forced_photometry = lightcurve['forced_photometry']
    forced_photometry = forced_photometry[forced_photometry['unit'] == 'diff_flux'].rename(columns={
        'brightness': 'fluxdiff_uJy',
        'e_brightness': 'fluxerrdiff_uJy',
    })

    lightcurve = pd.concat([detections, forced_photometry])
    lightcurve.sort_values(by='mjd', inplace=True)

    lightcurve = recover_mag_and_emag_with_signed_flux(lightcurve)
    lightcurve = lightcurve[['mjd', 'fid', 'mag', 'e_mag', 'isdiffpos']]
    return lightcurve

def get_feat_AO_to_InputDTO(tabular):
    ao_features = tabular['features'][["name", "fid", "value"]].copy()
    fid_map = {"g": "_1", "r": "_2", "g,r": "_12", None: ""}
    ao_features["name"] += ao_features["fid"].map(fid_map)
    ao_features = ao_features.sort_values("name")
    ao_features.replace({np.nan: None, np.inf: None, -np.inf: None}, inplace=True)
    oid_ao = query_ao_table(tabular['metadata'], "oid")
    feature_names = [f.replace("-", "_") for f in ao_features["name"].values]
    features_for_oid = dict(
        zip(feature_names, ao_features["value"].astype(np.double))
    )
    features = pd.DataFrame([features_for_oid], index=[oid_ao])
    features.index.name = 'oid'
    return features

def load_single_file(args):
    """
    Function to load a single file and return combined results for lightcurves and features.
    """
    path_chunk, dir_astro_features = args
    file_name = path_chunk.split("/")[-1]
    ao_lightcurves = pd.read_pickle(path_chunk)
    ao_features = pd.read_pickle(f'{dir_astro_features}/None_{file_name}') #None

    detections = []
    features = []
    for i in range(len(ao_lightcurves)): 
        aux_det, aux_feat = get_InputDTO(ao_lightcurves[i], ao_features[i])
        detections.append(aux_det)
        features.append(aux_feat)

    detections = pd.concat(detections)
    features = pd.concat(features)
    return detections, features

def load_astro_objects_as_InputDTO(args, use_multiprocessing=True, num_cores=None):
    """
    Load all astro_objects using multiprocessing.
    """

    if use_multiprocessing:
        num_cores = num_cores or cpu_count()

        logging.info(f"Loading data using multiprocessing with {num_cores} cores...")
        with Pool(processes=num_cores) as pool:
            results = list(tqdm.tqdm(pool.imap(load_single_file, args), total=len(args)))
    else:
        logging.info("Loading data sequentially...")
        results = [load_single_file(arg) for arg in tqdm.tqdm(args)]

    # Combine all results into single DataFrames
    all_detections = pd.concat([res[0] for res in results])
    all_features = pd.concat([res[1] for res in results])
    return all_detections, all_features

def get_subset_and_batches(all_detections, all_features, path_partition, subset, fold, batch_size):
    df_partitions = pd.read_parquet(path_partition)
    subset_fold = f'{subset}'
    if subset != 'test':
        subset_fold += f'_{fold}'
    ids_subset = df_partitions[df_partitions.partition == subset_fold].oid.values

    all_detections = all_detections.loc[ids_subset]
    all_features = all_features.loc[ids_subset]

    oid_batches = [
        ids_subset[i:i + batch_size]
        for i in range(0, len(ids_subset), batch_size)
    ]
    logging.info(f"Data divided into {len(oid_batches)} batches of size {batch_size}.")
    return all_detections, all_features, oid_batches