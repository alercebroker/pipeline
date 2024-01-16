import numpy as np
import pandas as pd
import time

from lc_classifier.features.preprocess import ElasticcPreprocessor
from lc_classifier.features import GPDRWExtractor
from lc_classifier.features.preprocess.preprocess_elasticc import shorten_lightcurve


folder = '/home/ireyes/Projects/random_forest_elasticc/mlp_round_2_v2_8_128_2048_updated_packages/'

lightcurves = pd.read_parquet(folder + 'test_lightcurves.parquet')
metadata = pd.read_parquet(folder + 'test_metadata.parquet')
labels = pd.read_parquet(folder + 'test_labels.parquet')

list_of_classes = [
    'AGN', 'CART', 'Cepheid', 'Delta Scuti', 'Dwarf Novae', 'EB', 'ILOT', 'KN',
    'M-dwarf Flare', 'PISN', 'RR Lyrae', 'SLSN', '91bg', 'Ia', 'Iax', 'Ib/c', 'II',
    'TDE', 'uLens'
]


def run_extractor_per_class(lightcurves, metadata):
    input_snids = ['26948852']

    class_metadata = metadata.loc[input_snids]
    preprocessor = ElasticcPreprocessor()
    feature_extractor = GPDRWExtractor(bands=['u', 'g', 'r', 'i', 'z', 'Y'])

    for n_days in [32, 64, 128, 256, 512, 1024, 2048]:
        class_lightcurves = lightcurves.loc[input_snids]
        class_lightcurves = class_lightcurves.groupby(level=0, group_keys=False).apply(
            lambda df: shorten_lightcurve(df, n_days))

        t0 = time.time()
        clean_lightcurves = preprocessor.preprocess(class_lightcurves)

        features = feature_extractor.compute_features(
            clean_lightcurves,
            metadata=class_metadata
        )
        tf = time.time() - t0
        print(f"len {len(clean_lightcurves)}, time {tf:.5f}")


run_extractor_per_class(lightcurves, metadata)
