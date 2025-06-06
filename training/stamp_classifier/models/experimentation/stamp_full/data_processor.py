import os
import time
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from datetime import datetime
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


#  normalization
def normalize_stamps(stamps_ndarray):
    new_shape = (
        stamps_ndarray.shape[0],
        stamps_ndarray.shape[1] * stamps_ndarray.shape[2],
        stamps_ndarray.shape[3])
    abs_ndarray = np.abs(stamps_ndarray.reshape(new_shape))
    abs_ndarray[np.isinf(abs_ndarray)] = np.nan
    abs_ndarray[abs_ndarray > 1e10] = np.nan
    is_infinite = (~np.isfinite(stamps_ndarray)).astype(float)

    maxval = np.nanpercentile(abs_ndarray, 99, axis=1)
    minval = np.nanmin(abs_ndarray, axis=1)
    
    stamps_ndarray = (stamps_ndarray - minval[:, np.newaxis, np.newaxis, :]) / (
        maxval[:, np.newaxis, np.newaxis, :] - minval[:, np.newaxis, np.newaxis, :] + 1e-8)

    stamps_ndarray = np.clip(stamps_ndarray, a_min=-2.0, a_max=2.0)

    stamps_ndarray = np.nan_to_num(stamps_ndarray, posinf=0.0, neginf=0.0)
    stamps_ndarray = np.concatenate([stamps_ndarray, is_infinite], axis=3)
    
    return stamps_ndarray


def normalize_batches(stamps_ndarray, batch_size):
    n_batches = int(np.ceil(len(stamps_ndarray) / batch_size))
    for batch_idx in range(n_batches):
        batch = stamps_ndarray[batch_idx*batch_size:(batch_idx+1)*batch_size, :, :, :3]
        stamps_ndarray[batch_idx*batch_size:(batch_idx+1)*batch_size] = normalize_stamps(batch)



def ra_dec_to_cartesian(ra: np.ndarray, dec: np.ndarray):
    """ Right ascension and declination to cartesian coordinates in a unit sphere"""
    ra = ra.flatten() / 180.0 * np.pi
    dec = dec.flatten() / 180.0 * np.pi
    x = np.cos(ra) * np.cos(dec)
    y = np.sin(ra) * np.cos(dec)
    z = np.sin(dec)

    return np.stack([x, y, z], axis=-1).astype(np.float32)

def extract_x_pos_y_from_df(df, oids):
    df = df.loc[oids]
    columns = ['science', 'reference', 'diff']
    ra = df['ra'].values
    dec = df['dec'].values

    position = ra_dec_to_cartesian(ra, dec)
    df_x = [df[c] for c in columns]

    x = np.stack([np.stack(data) for data in df_x])
    x = np.swapaxes(x, 0, 1)  # batch_size as 1st dim
    x = np.swapaxes(x, 1, 3)  # channel as last dim

    y = np.array(df['class'])

    # pos_vs = position[y == 'asteroid']
    # import matplotlib.pyplot as plt
    # from mpl_toolkits import mplot3d
    #
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    #
    # ax.scatter3D(pos_vs[:, 0], pos_vs[:, 1], pos_vs[:, 2], alpha=0.3)

    return x, position, y

def build_ndarrays(seed: int, path_load_data: str, path_save_data: str, date: str, has_avro: bool, satellite_as_bogus: bool):
    # Build suffix
    suffix = ''
    if has_avro:
        suffix += '_hasavro'
    if satellite_as_bogus:
        suffix += '_satbogus'

    # Load consolidated dataset
    if has_avro:
        consolidated_dataset_df = pd.read_pickle(os.path.join(path_load_data, 'consolidated_dataset_hasavro.pkl'))
    else:
        consolidated_dataset_df = pd.read_pickle(os.path.join(path_load_data, 'consolidated_dataset.pkl'))

    print('original shape', consolidated_dataset_df.shape)
    # 2869M

    # drop stamps that are not 63 x 63 pixels
    consolidated_dataset_df['s'] = consolidated_dataset_df['science'].apply(lambda x: x.shape)
    consolidated_dataset_df = consolidated_dataset_df[consolidated_dataset_df['s'] == (63, 63)]
    # 2876M

    # Ghost bogus have duplicates. We will drop them to avoid train-val-test info. leakage
    consolidated_dataset_df.drop_duplicates(subset=["oid"], keep='first', inplace=True)
    print('shape after drop_duplicates', consolidated_dataset_df.shape)

    if satellite_as_bogus:
        consolidated_dataset_df['class'] = consolidated_dataset_df['class'].replace('satellite', 'bogus')
    print(consolidated_dataset_df.groupby('class').count())

    # 2876M

    consolidated_dataset_df.set_index('oid', inplace=True)
    # 2876M

    oid_train_val, oid_test = train_test_split(
        consolidated_dataset_df.index.values,
        stratify=consolidated_dataset_df['class'].values, test_size=0.2, random_state=seed)
    # 2877M

    df_train_val = consolidated_dataset_df.loc[oid_train_val]
    oid_train, oid_val = train_test_split(
        df_train_val.index.values,
        stratify=df_train_val['class'].values, train_size=0.75, random_state=seed)

    x_train, pos_train, y_train = extract_x_pos_y_from_df(consolidated_dataset_df, oid_train)
    x_val, pos_val, y_val = extract_x_pos_y_from_df(consolidated_dataset_df, oid_val)
    x_test, pos_test, y_test = extract_x_pos_y_from_df(consolidated_dataset_df, oid_test)

    # Para cada split, obtener el candid correspondiente al oid
    candid_train = consolidated_dataset_df.loc[oid_train]['candid'].values
    candid_val = consolidated_dataset_df.loc[oid_val]['candid'].values
    candid_test = consolidated_dataset_df.loc[oid_test]['candid'].values

    d = {
        'x_train': x_train,
        'pos_train': pos_train,
        'y_train': y_train,
        'oid_train': oid_train,
        'candid_train': candid_train,

        'x_val': x_val,
        'pos_val': pos_val,
        'y_val': y_val,
        'oid_val': oid_val, 
        'candid_val': candid_val,

        'x_test': x_test,
        'pos_test': pos_test,
        'y_test': y_test,
        'oid_test': oid_test,
        'candid_test': candid_test,
    }


    #with open('data/test_first_stamps_fixed_ndarrays.pkl', 'wb') as f:
    # Save
    filename = f'consolidated_ndarrays{suffix}_{date}.pkl'
    with open(os.path.join(path_save_data, filename), 'wb') as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved: {filename}")



def process_ndarrays(path_save_data: str, date: str, has_avro: bool, satellite_as_bogus: bool):
    # Build suffix
    suffix = ''
    if has_avro:
        suffix += '_hasavro'
    if satellite_as_bogus:
        suffix += '_satbogus'

    # Load consolidated_ndarrays
    filename = f'consolidated_ndarrays{suffix}_{date}.pkl'
    with open(os.path.join(path_save_data, filename), 'rb') as f:
        d = pickle.load(f)

    x_train = d['x_train']
    pos_train = d['pos_train']
    y_train = d['y_train']
    oid_train = d['oid_train']
    candid_train = d['candid_train']

    x_val = d['x_val']
    pos_val = d['pos_val']
    y_val = d['y_val']
    oid_val = d['oid_val']
    candid_val = d['candid_val']

    x_test = d['x_test']
    pos_test = d['pos_test']
    y_test = d['y_test']
    oid_test = d['oid_test']
    candid_test = d['candid_test']

    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train).astype(np.int32)
    y_val = label_encoder.transform(y_val).astype(np.int32)
    y_test = label_encoder.transform(y_test).astype(np.int32)

    t0 = time.time()

    print('Normalizing training set')
    x_train = np.concatenate([x_train, np.zeros(shape=x_train.shape, dtype=np.float32)], axis=-1)
    normalize_batches(x_train, 1000)
    print('Elapsed time', time.time() - t0, '[s]')

    print('Normalizing validation set')
    x_val = np.concatenate([x_val, np.zeros(shape=x_val.shape, dtype=np.float32)], axis=-1)
    normalize_batches(x_val, 1000)
    print('Elapsed time', time.time() - t0, '[s]')

    print('Normalizing test set')
    x_test = np.concatenate([x_test, np.zeros(shape=x_test.shape, dtype=np.float32)], axis=-1)
    normalize_batches(x_test, 1000)
    print('Elapsed time', time.time() - t0, '[s]')

    d = {
        'x_train': x_train,
        'pos_train': pos_train,
        'y_train': y_train,
        'oid_train': oid_train,
        'candid_train': candid_train,

        'x_val': x_val,
        'pos_val': pos_val,
        'y_val': y_val,
        'oid_val': oid_val,
        'candid_val': candid_val,

        'x_test': x_test,
        'pos_test': pos_test,
        'y_test': y_test,
        'oid_test': oid_test,
        'candid_test': candid_test,

        'label_encoder': label_encoder
    }

    #with open('data/test_first_stamps_fixed_normalized_ndarrays.pkl', 'wb') as f:
    # Save
    filename = f'normalized_ndarrays{suffix}_{date}.pkl'
    with open(os.path.join(path_save_data, filename), 'wb') as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved: {filename}")


def dataset_as_png(output_folder, start, end):
    nlevels = 4
    df = pd.read_pickle(
        os.path.join(
            f'data/ML_nlevels{nlevels}_stamp_training.pkl'))
    print('original shape', df.shape)

    df.drop_duplicates(subset=["oid"], keep='first', inplace=True)
    print('shape after drop_duplicates', df.shape)
    df.sort_values('oid', inplace=True)

    df.set_index("oid", inplace=True)

    df = df.iloc[start:end].copy()

    # hard-coded hack used once
    # df = df.iloc[start:end]
    # df = df[df.astro_class == 'bogus'].copy()

    print(np.unique(df['class'].values, return_counts=True))

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    for index, row in tqdm(df.iterrows()):
        save_stamp_as_png(row, output_folder)


def save_stamp_as_png(stamp_series, output_folder):
    oid = stamp_series.name

    science = stamp_series.science
    reference = stamp_series.reference
    diff = stamp_series['diff']
    astro_class = stamp_series['class']

    fig, axs = plt.subplots(1, 3, figsize=(8, 4))
    fig.set_facecolor('white')

    fig.suptitle(f'{oid}, {astro_class}')

    axs[0].imshow(science)
    axs[0].axis('off')
    axs[0].set_title('science')

    axs[1].imshow(reference)
    axs[1].axis('off')
    axs[1].set_title('reference')

    axs[2].imshow(diff)
    axs[2].axis('off')
    axs[2].set_title('diff')

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            output_folder,
            f'{oid}.png'
        )
    )
    fig.clf()
    plt.close('all')


if __name__ == '__main__':

    path_load_data = '../../../data_acquisition/data/processed'
    path_save_data = './data'
    has_avro = False
    satellite_as_bogus = False

    date = datetime.now().strftime("%Y-%m-%d_%H-%M")

    os.makedirs(path_save_data, exist_ok=True)
    build_ndarrays(0, path_load_data, path_save_data, date, has_avro, satellite_as_bogus)
    process_ndarrays(path_save_data, date, has_avro, satellite_as_bogus)
    #crop_stamps(16)
    #low_res_stamps(4)
    #build_multiscale()

    # get_tf_datasets('full', batch_size=256)
    # get_datasets(0, 256)
    # dataset_as_png('stamps', 2000, 29000)