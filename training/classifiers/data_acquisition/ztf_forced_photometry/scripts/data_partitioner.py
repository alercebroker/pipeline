import os
import glob
import multiprocessing
import pandas as pd
import numpy as np
import tqdm
import logging
from sklearn.model_selection import train_test_split, StratifiedKFold

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
)

def extract_info(path_chunk):
    astro_objects_batch = pd.read_pickle(path_chunk)
    list_ids = [astro_object.get('detections').index[0] for astro_object in astro_objects_batch]
    return list_ids

def create_partitions(objects, dir_save_partition, num_folds):
    logging.info("Starting partition creation")
    objects = objects.reset_index()
    objects.rename(columns={"alerceclass": "class_name"}, inplace=True)

    X = objects['oid']
    y = objects["class_name"]


    #_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    ######################
    partitions = pd.read_parquet('preprocessed/partitions/241209/partitions.parquet')
    X_test = partitions[partitions.partition == 'test']
    ids_test = X_test[X_test.oid.isin(objects.oid)].oid.values

    X_test = X[X.isin(ids_test)]
    y_test = y.loc[X_test.index]
    ######################

    logging.info(f"Test set split completed, size: {len(X_test)}")
    test_indices = X_test.index
    test_set = objects.loc[test_indices].copy()
    test_set["partition"] = "test"

    training_validation = objects.drop(test_indices)
    kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)

    training_set_list = []
    validation_set_list = []
    n_samples = len(training_validation)

    for i, (training_index, validation_index) in enumerate(kf.split(np.zeros(n_samples), training_validation["class_name"])):
        training_set = training_validation.iloc[training_index].copy()
        validation_set = training_validation.iloc[validation_index].copy()
        training_set["partition"] = f"training_{i}"
        validation_set["partition"] = f"validation_{i}"
        training_set_list.append(training_set)
        validation_set_list.append(validation_set)
        logging.info(f"Fold {i}: Training set size: {len(training_set)}, Validation set size: {len(validation_set)}")

    partitions = pd.concat([test_set] + training_set_list + validation_set_list, axis=0)

    blind_test = test_set.copy()
    blind_test["class_name"] = "unknown"
    blind_partitions = pd.concat([blind_test] + training_set_list + validation_set_list, axis=0)

    os.makedirs(dir_save_partition, exist_ok=True)
    partitions.to_parquet(f"{dir_save_partition}/partitions.parquet")
    blind_partitions.to_parquet(f"{dir_save_partition}/blind_partitions.parquet")
    logging.info("Partitions saved successfully")

def run(path_all_objects, dir_astro_lightcurves, dir_save_partition, num_folds, num_cores=4):
    logging.info("Reading all objects")
    objects = pd.read_parquet(path_all_objects)
    logging.info(f"Total number of objects: {len(objects)}")
    path_chunks = glob.glob(f"{dir_astro_lightcurves}/astro_objects_batch_*.pkl")

    logging.info(f"Found {len(path_chunks)} chunks to process")
    with multiprocessing.Pool(processes=num_cores) as pool:
        list_ids = list(tqdm.tqdm(pool.imap_unordered(extract_info, path_chunks), total=len(path_chunks)))

    ids = np.concatenate(list_ids)
    objects = objects[objects.index.isin(ids)]

    logging.info(f"Filtered objects to {len(objects)} relevant entries")
    create_partitions(objects, dir_save_partition, num_folds)

if __name__ == "__main__":
    num_folds = 5
    path_all_objects = './raw/objects.parquet'

    ROOT = './preprocessed'
    dir_astro_lightcurves = f"{ROOT}/data_241209_ndetge8_ao"
    dir_save_partition = f"{ROOT}/partitions/241209_ndetge8"
    num_cores = 20

    run(path_all_objects, dir_astro_lightcurves, dir_save_partition, num_folds, num_cores)
