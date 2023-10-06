import os
import pandas as pd
from lc_classifier.features.core.base import AstroObject


def get_ztf_example() -> AstroObject:
    detections = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            'ZTF18abombrp_20231006/detections.csv'
        )
    )
    non_detections = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            'ZTF18abombrp_20231006/non_detections.csv'
        )
    )

    fid_map = {
        1: 'g',
        2: 'r',
        3: 'i'
    }
    detections = detections[
        [
            'candid',
            'tid',
            'mjd',
            'fid',
            'pid',
            'ra',
            'dec',
            'magpsf_corr',
            'sigmapsf_corr',
        ]
    ]
    detections.rename(columns={
        'magpsf_corr': 'brightness',
        'sigmapsf_corr': 'e_brightness'
    }, inplace=True)
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
            ["oid", "ZTF18abombrp"]
        ],
        columns=["field", "value"]
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
            'elasticc/elasticc_snid_15415799_detections.csv'
        )
    )
    metadata = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            'elasticc/elasticc_snid_15415799_metadata.csv'
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

    metadata = metadata[['field', 'value']]

    detections['ra'] = metadata[metadata['field'] == 'RA']['value'].values[0]
    detections['dec'] = metadata[metadata['field'] == 'DEC']['value'].values[0]
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
