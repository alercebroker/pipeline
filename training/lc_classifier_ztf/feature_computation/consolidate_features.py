import pandas as pd
import os
from tqdm import tqdm
from lc_classifier.utils import all_features_from_astro_objects
from lc_classifier.features.core.base import astro_object_from_dict


def get_shorten(filename: str):
    possible_n_days = filename.split("_")[-2]
    return possible_n_days


if __name__ == "__main__":
    dir_name = "data_231206_features"
    data_dir = os.listdir(dir_name)
    data_dir = [filename for filename in data_dir if "astro_objects_batch" in filename]
    data_dir = sorted(data_dir)

    all_features = []
    for batch_filename in tqdm(data_dir):
        full_filename = os.path.join(dir_name, batch_filename)
        shorten = get_shorten(full_filename)
        astro_objects_batch = pd.read_pickle(full_filename)
        astro_objects_batch = [astro_object_from_dict(ao) for ao in astro_objects_batch]
        features_batch = all_features_from_astro_objects(astro_objects_batch)
        features_batch["shorten"] = shorten
        all_features.append(features_batch)

    all_features = pd.concat(all_features, axis=0)
    print(all_features)
    all_features.to_parquet(os.path.join(dir_name, "consolidated_features.parquet"))
