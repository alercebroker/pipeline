import pandas as pd
import os
import multiprocessing as mp
from tqdm import tqdm
from lc_classifier.utils import all_features_from_astro_objects
from lc_classifier.features.core.base import astro_object_from_dict

def get_shorten(filename: str):
    return filename.split("/")[-1].split("_")[0]

def process_batch(batch_filename):
    """Function to process a single batch."""
    dir_name = "data_241209_ndetge8_ao_shorten_features"
    full_filename = os.path.join(dir_name, batch_filename)
    shorten = get_shorten(full_filename)
    astro_objects_batch = pd.read_pickle(full_filename)
    astro_objects_batch = [astro_object_from_dict(ao) for ao in astro_objects_batch]
    features_batch = all_features_from_astro_objects(astro_objects_batch)
    features_batch["shorten"] = shorten
    return features_batch

def main(num_workers):
    dir_name = "data_241209_ndetge8_ao_shorten_features"
    data_dir = os.listdir(dir_name)
    data_dir = [filename for filename in data_dir if "astro_objects_batch" in filename]
    data_dir = sorted(data_dir)
    
    num_workers = min(mp.cpu_count(), num_workers)  # Use available CPU cores
    with mp.Pool(processes=num_workers) as pool:
        all_features = list(tqdm(pool.imap(process_batch, data_dir), total=len(data_dir)))
    
    all_features = pd.concat(all_features, axis=0)
    object_columns = all_features.select_dtypes(include=["object"]).columns
    columns_to_convert = object_columns.drop("shorten", errors="ignore")
    all_features[columns_to_convert] = all_features[columns_to_convert].apply(pd.to_numeric, errors="coerce")

    all_features.to_parquet(os.path.join(dir_name, "consolidated_features.parquet"))

if __name__ == "__main__":
    num_workers = 20
    main(num_workers)
