import pandas as pd
import os
from sklearn.model_selection import train_test_split
from build_dataset import download_stamps


def get_test_oids():
    seed = 0
    # Replicate what was done when building the training set
    consolidated_dataset_df_filename = os.path.join(
        os.path.dirname(__file__),
        'data/consolidated_dataset.pkl')
    consolidated_dataset_df = pd.read_pickle(consolidated_dataset_df_filename)

    # drop stamps that are not 63 x 63 pixels
    consolidated_dataset_df['s'] = consolidated_dataset_df['science'].apply(lambda x: x.shape)
    consolidated_dataset_df = consolidated_dataset_df[consolidated_dataset_df['s'] == (63, 63)]

    # Ghost bogus have duplicates. We will drop them to avoid train-val-test info. leakage
    consolidated_dataset_df.drop_duplicates(subset=["oid"], keep='first', inplace=True)
    consolidated_dataset_df.set_index('oid', inplace=True)

    oid_train_val, oid_test = train_test_split(
        consolidated_dataset_df.index.values,
        stratify=consolidated_dataset_df['class'].values, test_size=0.2, random_state=seed)

    test_dataset = consolidated_dataset_df.loc[oid_test]
    return test_dataset


if __name__ == '__main__':
    test_dataset = get_test_oids()
    ghosts = test_dataset[test_dataset['candid'] == 'unknown']
    non_ghost = test_dataset[test_dataset['candid'] != 'unknown']
    non_ghost_oids = non_ghost.index.values

    download_stamps(non_ghost_oids, 'test_first_stamps_download.pkl', strategy='first')

    downloaded_stamps = pd.read_pickle('test_first_stamps_download.pkl')
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

    test_first_detection_dataset.to_pickle('test_first_stamps_dataset.pkl')
