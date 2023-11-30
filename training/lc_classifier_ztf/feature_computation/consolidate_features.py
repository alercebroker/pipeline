import pandas as pd
import os
from tqdm import tqdm
from lc_classifier.classifiers.base import all_features_from_astro_objects


dir_name = 'data_231130'
data_dir = os.listdir(dir_name)
data_dir = [filename for filename in data_dir if 'astro_objects_batch' in filename]
data_dir = sorted(data_dir)

all_features = []
for batch_filename in tqdm(data_dir):
    full_filename = os.path.join(dir_name, batch_filename)
    astro_objects_batch = pd.read_pickle(full_filename)
    features_batch = all_features_from_astro_objects(astro_objects_batch)
    all_features.append(features_batch)

all_features = pd.concat(all_features, axis=0)
print(all_features)
all_features.to_parquet(os.path.join(dir_name, 'consolidated_features.parquet'))
