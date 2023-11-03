import numpy as np
import os
import pandas as pd
from lc_classifier.features.core.base import AstroObject
from typing import List


def get_ztf_example(index: int) -> AstroObject:
    folders = [
        'ZTF18abombrp_20231006',
        'ZTF18aasycma_20231012',
        'ZTF19aawgxdn_20231025'
    ]
    assert index < len(folders)

    folder = folders[index]
    detections = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            folder,
            'detections.csv'
        )
    )
    non_detections = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            folder,
            'non_detections.csv'
        )
    )

    fid_map = {
        1: 'g',
        2: 'r',
        3: 'i'
    }

    is_corrected = True in detections['corrected'].unique()
    if is_corrected:
        detections.rename(columns={
            'magpsf_corr': 'brightness',
            'sigmapsf_corr': 'e_brightness'
        }, inplace=True)
    else:
        detections.rename(columns={
            'magpsf': 'brightness',
            'sigmapsf': 'e_brightness'
        }, inplace=True)
    detections = detections[
        [
            'candid',
            'tid',
            'mjd',
            'fid',
            'pid',
            'ra',
            'dec',
            'brightness',
            'e_brightness',
        ]
    ]
    detections = detections.dropna(subset=['brightness'])
    detections['sid'] = 'ztf'
    detections['unit'] = 'magnitude'
    detections['fid'] = detections['fid'].map(fid_map)

    non_detections = non_detections[
        [
            'tid',
            'mjd',
            'fid',
            'diffmaglim',
        ]
    ]
    non_detections.rename(columns={
        'diffmaglim': 'brightness'}, inplace=True)

    non_detections['sid'] = 'ztf'
    non_detections['unit'] = 'magnitude'
    non_detections['fid'] = non_detections['fid'].map(fid_map)

    metadata = pd.DataFrame(
        [
            ["aid", "aid_example"],
            ["oid", folder.split('_')[0]]
        ],
        columns=["name", "value"]
    )

    astro_object = AstroObject(
        detections=detections,
        non_detections=non_detections,
        metadata=metadata
    )
    return astro_object


def get_elasticc_example() -> AstroObject:
    detections = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            'elasticc/elasticc_snid_53125011_detections.csv'
        )
    )
    metadata = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            'elasticc/elasticc_snid_53125011_metadata.csv'
        )
    )

    detections = detections[
        [
            'MJD',
            'BAND',
            'FLUXCAL',
            'FLUXCALERR',
            'PHOTFLAG'
        ]
    ]
    detections.rename(columns={
        'MJD': 'mjd',
        'BAND': 'fid',
        'FLUXCAL': 'brightness',
        'FLUXCALERR': 'e_brightness'
    }, inplace=True)
    detections['candid'] = None
    detections['tid'] = 'elasticc_telescope'
    detections['sid'] = 'elasticc_survey'
    detections['pid'] = 'elasticc_program'

    metadata = metadata[['name', 'value']]

    detections['ra'] = float(metadata[metadata['name'] == 'RA']['value'].values[0])
    detections['dec'] = float(metadata[metadata['name'] == 'DEC']['value'].values[0])
    detections['unit'] = 'diff_flux'

    is_detected = detections['PHOTFLAG'] > 0
    detections.drop(columns=['PHOTFLAG'], inplace=True)
    forced_photometry = detections[~is_detected]
    detections = detections[is_detected]

    metadata.loc[len(metadata)] = ['aid', 'aid_example']

    astro_object = AstroObject(
        detections=detections,
        forced_photometry=forced_photometry,
        metadata=metadata
    )
    return astro_object


def get_elasticc_example_2() -> AstroObject:
    detections = pd.read_pickle(
        os.path.join(
            os.path.dirname(__file__),
            'elasticc_lc_38135580.pkl'
        )
    )

    detections = detections[
        [
            'MJD',
            'BAND',
            'FLUXCAL',
            'FLUXCALERR',
            'PHOTFLAG'
        ]
    ].copy()
    detections.rename(columns={
        'MJD': 'mjd',
        'BAND': 'fid',
        'FLUXCAL': 'brightness',
        'FLUXCALERR': 'e_brightness'
    }, inplace=True)
    detections['candid'] = None
    detections['tid'] = 'elasticc_telescope'
    detections['sid'] = 'elasticc_survey'
    detections['pid'] = 'elasticc_program'
    detections['unit'] = 'diff_flux'
    detections['ra'] = np.nan
    detections['dec'] = np.nan

    is_detected = detections['PHOTFLAG'] > 0
    detections.drop(columns=['PHOTFLAG'], inplace=True)
    forced_photometry = detections[~is_detected]
    detections = detections[is_detected]

    metadata = pd.DataFrame(
        data=[('aid', detections.index.values[0])],
        columns=['name', 'value']
    )

    astro_object = AstroObject(
        detections=detections,
        forced_photometry=forced_photometry,
        metadata=metadata
    )
    return astro_object


def get_ztf_forced_training_examples() -> List[AstroObject]:
    ztf_data = pd.read_parquet(
        os.path.join(
            os.path.dirname(__file__),
            'ztf_forced_examples_training.parquet'
        )
    )
    print(ztf_data.iloc[0])

    def df_to_astro_object(df):
        df['detected'] = np.abs(df['forcediffimsnr']) > 5.0
        diff_flux = df[[
            'index', 'forcediffimflux',
            'forcediffimfluxunc', 'fid',
            'mjd', 'detected'
        ]].copy()
        diff_flux.rename(columns={
            'forcediffimflux': 'brightness',
            'forcediffimfluxunc': 'e_brightness'
        }, inplace=True)
        diff_flux['unit'] = 'diff_flux'

        magnitude = df[[
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
        df['candid'] = np.nan
        df['pid'] = 'ztf_forced_phot_service'

        # TODO: complete this later
        df['ra'] = np.nan
        df['dec'] = np.nan

        metadata = pd.DataFrame(
            [
                ["aid", "aid_"+df.index.values[0]],
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
    
    oids = ztf_data.index.unique()
    astro_objects = []
    for oid in oids:
        astro_object = df_to_astro_object(ztf_data.loc[oid].copy())
        astro_objects.append(astro_object)

    return astro_objects


if __name__ == '__main__':
    astro_objects = get_ztf_forced_training_examples()
    print(astro_objects[0])
