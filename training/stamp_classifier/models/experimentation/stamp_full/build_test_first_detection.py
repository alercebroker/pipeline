import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append('../../../data_acquisition')

from build_dataset import download_stamps


def get_test_oids(path_load_data: str, path_data: str):
    if os.path.exists(f'{path_data}/normalized_ndarrays.pkl'):
        normalized_ndarrays = pd.read_pickle(f'{path_data}/normalized_ndarrays.pkl')
        oid_test = normalized_ndarrays['oid_test']
    else:
        raise ValueError(f'{path_data}/normalized_ndarrays.pkl not found')

    # Replicate what was done when building the training set
    consolidated_dataset_df_filename = os.path.join(
        os.path.dirname(__file__),
        path_load_data,
        'consolidated_dataset.pkl'
        )
    consolidated_dataset_df = pd.read_pickle(consolidated_dataset_df_filename)

    # drop stamps that are not 63 x 63 pixels
    consolidated_dataset_df['s'] = consolidated_dataset_df['science'].apply(lambda x: x.shape)
    consolidated_dataset_df = consolidated_dataset_df[consolidated_dataset_df['s'] == (63, 63)]

    # Ghost bogus have duplicates. We will drop them to avoid train-val-test info. leakage
    consolidated_dataset_df.drop_duplicates(subset=["oid"], keep='first', inplace=True)
    consolidated_dataset_df.set_index('oid', inplace=True)

    test_dataset = consolidated_dataset_df.loc[oid_test]
    return test_dataset


if __name__ == '__main__':
    path_load_data = '../../../data_acquisition/data/processed'
    path_data = 'data'
    test_dataset = get_test_oids(path_load_data, path_data)
    ghosts = test_dataset[test_dataset['candid'] == 'unknown']
    non_ghost = test_dataset[test_dataset['candid'] != 'unknown']
    non_ghost_oids = non_ghost.index.values

    download_stamps(non_ghost_oids, f'{path_data}/test_first_stamps_download.pkl', strategy='first')

    downloaded_stamps = pd.read_pickle(f'{path_data}/test_first_stamps_download.pkl')
    downloaded_stamps.set_index('oid', inplace=True)
    downloaded_stamps['class'] = test_dataset.loc[downloaded_stamps.index.values]['class']

    fields = [
        'candid',
        'ra',
        'dec',
        'science',
        'reference',
        'diff',
        'class'
    ]

    test_first_detection_dataset = pd.concat([
        downloaded_stamps[fields],
        ghosts[fields]
    ], axis=0)

    test_first_detection_dataset.to_pickle(f'{path_data}/test_first_stamps_dataset.pkl')