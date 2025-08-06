import os 
import pickle

import numpy as np
import pandas as pd
import astropy.units as u

from astropy.coordinates import SkyCoord
from sklearn.preprocessing import QuantileTransformer

import logging
#logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


def process_coordinates(
    oids, ra: np.ndarray, dec: np.ndarray, coord_type: str = "cartesian"
) -> np.ndarray:
    """
    Dispatch function to process celestial coordinates.
    """
    if coord_type == "cartesian":
        return convert_to_cartesian(oids, ra, dec)
    elif coord_type == "spherical":
        return convert_to_spherical(oids, ra, dec)
    elif coord_type == "none":
        return None
    else:
        raise ValueError(
            f"Unknown coord_type: {coord_type}. Use 'cartesian', 'spherical', or 'none'."
        )

def convert_to_cartesian(oids, ra: np.ndarray, dec: np.ndarray):
    """Right ascension and declination to cartesian coordinates in a unit sphere"""
    ra = ra.flatten() / 180.0 * np.pi
    dec = dec.flatten() / 180.0 * np.pi
    x = np.cos(ra) * np.cos(dec)
    y = np.sin(ra) * np.cos(dec)
    z = np.sin(dec)

    return pd.DataFrame(
        np.stack([x, y, z], axis=-1).astype(np.float32),
        columns=["pos_x", "pos_y", "pos_z"],
        index=oids,
    )

def convert_to_spherical(oids, ra: np.ndarray, dec: np.ndarray):
    """Right ascension and declination to spherical coordinates"""
    coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg))
    gal_l = coords.galactic.l.value
    gal_b = coords.galactic.b.value
    ecl_long = coords.barycentricmeanecliptic.lon.value
    ecl_lat = coords.barycentricmeanecliptic.lat.value

    return pd.DataFrame(
        np.stack([gal_l, gal_b, ecl_long, ecl_lat], axis=-1).astype(np.float32),
        columns=["gal_l", "gal_b", "ecl_long", "ecl_lat"],
        index=oids,
    )


def apply_normalization(metadata, norm_type, dict_info_model, path_norm_dir, is_test_only):
    if is_test_only:
        logging.info("🔄 Loading normalization parameters for val/test...")
        dict_info_model.update(load_normalization_params(norm_type, path_norm_dir))
    else:
        logging.info("📊 Computing and saving normalization parameters for training...")
        compute_normalization_params(metadata, norm_type, dict_info_model)
        save_normalization_params(norm_type, dict_info_model, path_norm_dir)

    if norm_type == 'QT':
        return qt_normalization(metadata, dict_info_model['qt'])
    elif norm_type == 'z-score':
        return zscore_normalization(metadata, dict_info_model['norm_means'], dict_info_model['norm_stds'])
    else:
        raise ValueError(f"Tipo de normalización no soportado: {norm_type}")

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

def compute_normalization_params(metadata, norm_type, dict_info_model):
    mask_valid = ~(metadata == -999).any(axis=1)
    metadata_valid = metadata.loc[mask_valid].copy()

    if norm_type == 'QT':
        qt = QuantileTransformer(output_distribution='uniform', random_state=42, subsample=5000)
        qt.fit(metadata_valid)
        dict_info_model['qt'] = qt
    elif norm_type == 'z-score':
        means = metadata_valid.mean()
        stds = metadata_valid.std().replace(0, 1)
        dict_info_model['norm_means'] = means
        dict_info_model['norm_stds'] = stds

def save_normalization_params(norm_type, dict_info_model, path_dir):
    os.makedirs(path_dir, exist_ok=True)
    if norm_type == 'QT':
        with open(os.path.join(path_dir, 'quantile_transformer.pkl'), 'wb') as f:
            pickle.dump(dict_info_model['qt'], f)
    elif norm_type == 'z-score':
        with open(os.path.join(path_dir, 'z_score.pkl'), 'wb') as f:
            pickle.dump({
                'means': dict_info_model['norm_means'],
                'stds': dict_info_model['norm_stds']
            }, f)

def load_normalization_params(norm_type, path_dir):
    if norm_type == 'QT':
        with open(os.path.join(path_dir, 'quantile_transformer.pkl'), 'rb') as f:
            qt = pickle.load(f)
        return {'qt': qt}
    elif norm_type == 'z-score':
        with open(os.path.join(path_dir, 'z_score.pkl'), 'rb') as f:
            data = pickle.load(f)
        return {
            'norm_means': pd.Series(data['means']),
            'norm_stds': pd.Series(data['stds'])
        }

def fill_and_clipping_metadata(metadata):
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

    metadata = metadata.fillna(-999)
    metadata['non_detections'] = metadata['ncovhist'] - metadata['ndethist']
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