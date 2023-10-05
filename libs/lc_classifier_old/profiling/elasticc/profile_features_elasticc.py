import numpy as np
import pandas as pd
import time
import cProfile

from lc_classifier.features.preprocess import ElasticcPreprocessor
from lc_classifier.features import ElasticcFeatureExtractor
from lc_classifier.features.preprocess.preprocess_elasticc import shorten_lightcurve


folder = '/home/ireyes/Projects/random_forest_elasticc/mlp_round_2_v2_8_128_2048_updated_packages/'

n_days = 2048
lightcurves = pd.read_parquet(folder + 'test_lightcurves.parquet')
metadata = pd.read_parquet(folder + 'test_metadata.parquet')
labels = pd.read_parquet(folder + 'test_labels.parquet')

list_of_classes = [
    'AGN', 'CART', 'Cepheid', 'Delta Scuti', 'Dwarf Novae', 'EB', 'ILOT', 'KN',
    'M-dwarf Flare', 'PISN', 'RR Lyrae', 'SLSN', '91bg', 'Ia', 'Iax', 'Ib/c', 'II',
    'TDE', 'uLens'
]


def run_extractor_per_class(chosen_class, lightcurves, metadata):
    class_snids = labels.index.values
    # class_snids = labels[labels['label'] == chosen_class].index.values
    np.random.seed(1)
    input_snids = np.random.choice(class_snids, 50, replace=False)

    class_lightcurves = lightcurves.loc[input_snids]
    class_metadata = metadata.loc[input_snids]

    class_lightcurves = class_lightcurves.groupby(level=0, group_keys=False).apply(
        lambda df: shorten_lightcurve(df, n_days))

    preprocessor = ElasticcPreprocessor()
    feature_extractor = ElasticcFeatureExtractor(round=2)

    # call to compile jax functions
    clean_lightcurves = preprocessor.preprocess(class_lightcurves)
    features = feature_extractor.compute_features(
        clean_lightcurves,
        metadata=class_metadata,
        force_snids=input_snids
    )

    profiler = cProfile.Profile()
    profiler.enable()
    t0 = time.time()
    clean_lightcurves = preprocessor.preprocess(class_lightcurves)

    features = feature_extractor.compute_features(
        clean_lightcurves,
        metadata=class_metadata,
        force_snids=input_snids
    )
    tf = time.time() - t0
    profiler.disable()
    profiler.dump_stats(f'{chosen_class}_profile.prof')
    return tf / len(input_snids)


agn_time = run_extractor_per_class('all_classes', lightcurves, metadata)
print(agn_time)
exit()

avg_times = {}
np.random.shuffle(list_of_classes)
for chosen_class in list_of_classes:
    avg_times[chosen_class] = run_extractor_per_class(
        chosen_class, lightcurves, metadata)

print(avg_times)
