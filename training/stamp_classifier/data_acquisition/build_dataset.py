import pandas as pd
import numpy as np
import gzip
import io
import os

from astropy.io.fits import open as fits_open
from alerce.core import Alerce
from tqdm import tqdm


def extract_image_from_fits(stamp_byte, with_header=False):
    with gzip.open(io.BytesIO(stamp_byte), 'rb') as f:
        with fits_open(io.BytesIO(f.read()), ignore_missing_simple=True) as hdul:
            im = hdul[0].data
            header = hdul[0].header
    if with_header:
        return im, header
    else:
        return im


def extract_ghost_bogus():
    """ Look for all bogus from Javier's dataset that are not in ALeRCE's database.
    Then fetch those stamps from Esteban's dataset."""
    alerce_client = Alerce()
    dataset_javier = pd.read_pickle(
        'data/raw/training_set_Aug-07-2020-oid-class_jarredondo.pkl')
    bogus_javier = dataset_javier[dataset_javier['class'] == 'bogus'].drop_duplicates()

    bogus_javier_in_db_list = []
    n_batches = int(np.ceil(len(bogus_javier) / 100))
    for i in range(n_batches):
        oids = bogus_javier.oid.values[i * 100:(i + 1) * 100]
        objects = alerce_client.query_objects(
            oid=oids,
            format="pandas",
            page_size=100
        )
        bogus_javier_in_db_list.append(objects)
        print(len(objects), i * 100)
    bogus_javier_in_db = pd.concat(bogus_javier_in_db_list, axis=0)

    not_in_db = [o for o in bogus_javier.oid.values if o not in bogus_javier_in_db.oid.values]
    print(len(not_in_db), "oids from bogus are not in ALeRCE's database")

    dataset_ereyes = pd.read_pickle(
        'data/raw/training_set_Jun-22-2020.pkl')
    dataset_ereyes.set_index('oid', inplace=True)
    ghost_bogus_ereyes = dataset_ereyes.loc[not_in_db].copy()
    print('Esteban has', len(ghost_bogus_ereyes), 'stamps from ghost bogus')

    # Extract stamps from fits
    ghost_bogus_ereyes['science'] = ghost_bogus_ereyes['cutoutScience'].apply(
        lambda x: extract_image_from_fits(x, with_header=False))
    ghost_bogus_ereyes['reference'] = ghost_bogus_ereyes['cutoutTemplate'].apply(
        lambda x: extract_image_from_fits(x, with_header=False))
    ghost_bogus_ereyes['diff'] = ghost_bogus_ereyes['cutoutDifference'].apply(
        lambda x: extract_image_from_fits(x, with_header=False))

    ghost_bogus_ereyes.to_pickle('data/processed/ghost_bogus.pkl')


def curated_bogus_list():
    """ List of known bogus objects in ALeRCE's database."""
    training_set = pd.read_pickle(
        'data/raw/dfcrossmatches_ZTF_prioritized_v8.0.3.pickle')
    dataset_javier = pd.read_pickle(
        'data/raw/training_set_Aug-07-2020-oid-class_jarredondo.pkl')
    bogus_javier = dataset_javier[dataset_javier['class'] == 'bogus'].drop_duplicates()
    ghost_bogus = pd.read_pickle('data/processed/ghost_bogus.pkl')
    non_ghost_bogus = bogus_javier[~bogus_javier['oid'].isin(ghost_bogus.index.unique().values)]
    bogus_wo_astros = set(non_ghost_bogus.oid.values) - set(training_set.index.values)
    mislabeled_bogus = pd.read_csv(
        'data/raw/satellites/pearl_necklace_tagged_as_bogus.txt',
        comment='#', header=None, names=['oid']
    )
    curated_bogus = bogus_wo_astros - set(mislabeled_bogus.oid.values)
    curated_bogus = list(curated_bogus)

    # Keep only bogus with less than 10 detections in the database. Otherwise they could have been mislabeled.
    curated_bogus_metadata_list = []
    n_batches = int(np.ceil(len(curated_bogus) / 100))
    alerce_client = Alerce()
    for i in tqdm(range(n_batches)):
        oids = curated_bogus[i * 100:(i + 1) * 100]
        objects = alerce_client.query_objects(
            oid=oids,
            format="pandas",
            page_size=100
        )
        curated_bogus_metadata_list.append(objects)
    curated_bogus_metadata = pd.concat(curated_bogus_metadata_list, axis=0)
    curated_bogus = curated_bogus_metadata[curated_bogus_metadata['ndet'] < 10].oid.values

    # Add verified long bogus sources
    long_bogus = pd.read_csv(
        'data/raw/verified_long_bogus.txt', comment='#', header=None)
    long_bogus.rename(columns={0: 'oid'}, inplace=True)

    curated_bogus = pd.DataFrame(curated_bogus, columns=['oid'])
    curated_bogus = pd.concat([curated_bogus, long_bogus], axis=0)
    curated_bogus.to_csv(
        'data/processed/curated_bogus.txt', index=False, header=False)


def curated_asteroid_list():
    dataset_javier = pd.read_pickle(
        'data/raw/training_set_Aug-07-2020-oid-class_jarredondo.pkl')
    asteroids = dataset_javier[dataset_javier['class'] == 'asteroid']

    # check ndets of asteroid oids
    metadata_list = []
    n_batches = int(np.ceil(len(asteroids) / 100))
    alerce_client = Alerce()
    for i in tqdm(range(n_batches)):
        oids = asteroids.oid.values[i * 100:(i + 1) * 100]
        objects = alerce_client.query_objects(
            oid=oids,
            format="pandas",
            page_size=100
        )
        metadata_list.append(objects)
    metadata = pd.concat(metadata_list, axis=0)

    # keeping ndet == 1 is a safe bet
    single = metadata[metadata['ndet'] == 1]
    curated_asteroid = pd.DataFrame(single, columns=['oid'])
    curated_asteroid.to_csv(
        'data/raw/curated_asteroid.txt', index=False, header=False)


def download_bogus():
    curated_bogus = pd.read_csv('data/processed/curated_bogus.txt', header=None)
    download_stamps(curated_bogus[0].values, 'data/processed/database_bogus.pkl')


def download_sne():
    training_set = pd.read_pickle('data/raw/dfcrossmatches_ZTF_prioritized_v8.0.3.pickle')
    sne_classes = ['SNIa', 'SNII', 'SNIbc', 'SNIIn', 'SLSN', 'SNIIb']
    sne_list = training_set[training_set['classALeRCE'].isin(sne_classes)]
    download_stamps(sne_list.index.unique().values, 'data/processed/sne_stamps.pkl')


def download_vs():
    training_set = pd.read_pickle('data/raw/dfcrossmatches_ZTF_prioritized_v8.0.3.pickle')
    vs_classes = [
        'EB/EW', 'LPV', 'RRL', 'RSCVn', 'YSO',
        'EA', 'DSCT', 'Ceph', 'Periodic-Other', 'ZZ']
    vs_list = training_set[training_set['classALeRCE'].isin(vs_classes)]
    vs_sample = vs_list.groupby('classALeRCE', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 1200), random_state=0))
    download_stamps(vs_sample.index.unique().values, 'data/processed/vs_stamps.pkl')


def download_agn():
    training_set = pd.read_pickle('data/raw/dfcrossmatches_ZTF_prioritized_v8.0.3.pickle')
    agn_classes = ['AGN', 'Blazar', 'QSO']
    agn_list = training_set[training_set['classALeRCE'].isin(agn_classes)]
    agn_sample = agn_list.groupby('classALeRCE', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 4000), random_state=0))
    download_stamps(agn_sample.index.unique().values, 'data/processed/agn_stamps.pkl')


def download_asteroid():
    curated_asteroid = pd.read_csv('data/raw/curated_asteroid.txt', header=None)
    download_stamps(curated_asteroid[0].values, 'data/processed/asteroid_stamps.pkl')


def download_stamps(oid_list, output_file, strategy='random'):
    """ For each oid it chooses a random detection with stamp and download it."""
    downloaded_stamps = []
    if os.path.exists(output_file):
        loaded_checkpoint = pd.read_pickle(output_file)
        downloaded_stamps.append(loaded_checkpoint)
        oid_list = set(oid_list) - set(loaded_checkpoint.oid.values)
        oid_list = list(oid_list)

    alerce_client = Alerce()
    for i, oid in enumerate(tqdm(oid_list)):
        detections = alerce_client.search_client.query_detections(oid, format='pandas')
        if strategy == 'random':
            detections_with_stamps = detections[detections.has_stamp]
            detection = detections_with_stamps.sample()
            candid = detection['candid'].values[0]
            stamps = alerce_client.get_stamps(
                oid,
                candid=candid,
                format='numpy'
            )
            science, reference, diff = stamps
        elif strategy == 'first':
            success = False
            detections.sort_values('mjd', inplace=True)

            n_dets = len(detections)
            for i in range(n_dets):
                detection = detections.iloc[[i]].copy()
                candid = detection['candid'].values[0]
                try:
                    stamps = alerce_client.get_stamps(
                        oid,
                        candid=candid,
                        format='numpy'
                    )
                    science, reference, diff = stamps
                    if science.shape == (63, 63):
                        success = True
                        break
                except Exception as e:
                    pass

            if not success:
                print(f'[{oid}] No valid stamp found\n')
                continue
        else:
            raise ValueError('Invalid value for argument "strategy"')

        detection['oid'] = oid
        detection['science'] = [science]
        detection['reference'] = [reference]
        detection['diff'] = [diff]

        detection = detection[[
            'oid',
            'candid',
            'ra',
            'dec',
            'science',
            'reference',
            'diff',
        ]]
        downloaded_stamps.append(detection)
        if (i % 100) == 0:
            downloaded_stamps_df = pd.concat(downloaded_stamps, axis=0)
            downloaded_stamps_df.to_pickle(output_file)

    downloaded_stamps_df = pd.concat(downloaded_stamps, axis=0)
    downloaded_stamps_df.to_pickle(output_file)


def download_satellites():
    satellites = pd.read_csv('data/raw/satellites/bogus_moving_v1.0.0.csv')
    mislabeled_bogus = pd.read_csv(
        'data/raw/satellites/pearl_necklace_tagged_as_bogus.txt',
        comment='#', header=None, names=['oid']
    )
    download_stamps(
        np.concatenate([
            satellites.oid.values,
            mislabeled_bogus.oid.values]),
        'data/processed/satellites.pkl'
    )


def consolidate_datasets():
    sne = pd.read_pickle('data/processed/sne_stamps.pkl')
    sne['class'] = 'sn'

    vs = pd.read_pickle('data/processed/vs_stamps.pkl')
    vs['class'] = 'vs'

    agn = pd.read_pickle('data/processed/agn_stamps.pkl')
    agn['class'] = 'agn'

    ghost_bogus = pd.read_pickle('data/processed/ghost_bogus.pkl')
    ghost_bogus.reset_index(inplace=True)
    ghost_bogus['candid'] = 'unknown'

    database_bogus = pd.read_pickle('data/processed/database_bogus.pkl')
    database_bogus['class'] = 'bogus'

    asteroid = pd.read_pickle('data/processed/asteroid_stamps.pkl')
    asteroid['class'] = 'asteroid'

    satellite = pd.read_pickle('data/processed/satellites.pkl')
    satellite['class'] = 'satellite'

    key_fields = [
        'oid',
        'candid',
        'ra',
        'dec',
        'science',
        'reference',
        'diff',
        'class'
    ]
    sne = sne[key_fields]
    vs = vs[key_fields]
    agn = agn[key_fields]
    ghost_bogus = ghost_bogus[key_fields]
    database_bogus = database_bogus[key_fields]
    asteroid = asteroid[key_fields]
    satellite = satellite[key_fields]

    dataset = pd.concat([
        sne,
        vs,
        agn,
        ghost_bogus,
        database_bogus,
        asteroid,
        satellite
    ], axis=0)
    dataset.to_pickle('data/processed/consolidated_dataset.pkl')


if __name__ == "__main__":
    # First you need the raw data
    # then this script will create processed data
    os.makedirs('data/processed')

    extract_ghost_bogus()
    curated_bogus_list()
    curated_asteroid_list()

    download_sne()
    download_vs()
    download_agn()
    download_bogus()
    download_asteroid()
    download_satellites()

    consolidate_datasets()
