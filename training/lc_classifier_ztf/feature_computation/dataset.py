import pickle

import numpy as np
import pandas as pd
from lc_classifier.base import AstroObject
from tqdm import tqdm
from typing import List


class NoDetections(Exception):
    pass


def create_astro_object(
        lc_df: pd.DataFrame,
        object_info: pd.DataFrame) -> AstroObject:

    lc_df = lc_df.copy()
    # lc_df['detected'] = np.abs(lc_df['flux_diff_ujy'] / lc_df['sigma_flux_diff_ujy']) > 5

    lc_df['fid'] = lc_df['fid'].map({
        1: 'g',
        2: 'r',
        3: 'i'
    })
    lc_df = lc_df[lc_df['fid'].isin(['g', 'r'])]

    if len(lc_df[lc_df['detected']]) == 0:
        raise NoDetections()

    diff_flux = lc_df[[
        'index', 'flux_diff_ujy',
        'sigma_flux_diff_ujy', 'fid',
        'mjd', 'detected'
    ]].copy()

    diff_flux.rename(columns={
        'flux_diff_ujy': 'brightness',
        'sigma_flux_diff_ujy': 'e_brightness'
    }, inplace=True)
    diff_flux['unit'] = 'diff_flux'

    magnitude = lc_df[[
        'index', 'mag_tot',
        'sigma_mag_tot', 'fid',
        'mjd', 'detected'
    ]].copy()
    magnitude.rename(columns={
        'mag_tot': 'brightness',
        'sigma_mag_tot': 'e_brightness'
    }, inplace=True)
    magnitude['unit'] = 'magnitude'

    df = pd.concat([diff_flux, magnitude], axis=0)
    df['sid'] = 'ztf'
    df['tid'] = 'ztf_telescope'
    df.rename(columns={'index': 'candid'}, inplace=True)
    df['pid'] = 'ztf_forced_phot_service'

    df['ra'] = object_info['ra']
    df['dec'] = object_info['dec']

    df.dropna(subset=['brightness', 'e_brightness'], inplace=True)

    first_detected_mjd = np.min(df[df['detected']]['mjd'])
    last_detected_mjd = np.max(df[df['detected']]['mjd'])

    df = df[df['mjd'] <= last_detected_mjd]
    df = df[df['mjd'] >= (first_detected_mjd - 30)]

    metadata = pd.DataFrame(
        [
            ["aid", "aid_" + df.index.values[0]],
            ["oid", df.index.values[0]]
        ],
        columns=["name", "value"]
    )

    astro_object = AstroObject(
        detections=df[df['detected']],
        forced_photometry=df[~df['detected']],
        metadata=metadata
    )
    return astro_object


def save_batch(astro_objects: List[AstroObject], batch_id: int):
    filename = f'data_231128/astro_objects_batch_{batch_id:04}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(astro_objects, f)


if __name__ == '__main__':
    # Build AstroObjects

    lightcurves = pd.read_parquet('data_231128/lightcurves_231128.parquet')
    object_df = pd.read_parquet('data_231128/objects_231128.parquet')

    lightcurves.set_index('oid', inplace=True)
    # object_df.set_index('oid', inplace=True)

    oids = np.unique(object_df.index.values)

    astro_objects_list = []
    for oid in tqdm(oids):
        lc = lightcurves.loc[[oid]]
        object_info = object_df.loc[oid]
        try:
            astro_object = create_astro_object(lc, object_info)
            # plot_astro_object(astro_object, unit='diff_flux')
            astro_objects_list.append(astro_object)
        except NoDetections:
            print(object_info)
            print('Object with no detections')

    save_batch(astro_objects_list, 0)
