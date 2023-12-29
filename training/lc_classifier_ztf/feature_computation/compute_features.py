import numpy as np
import pandas as pd
import os
from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor, ShortenPreprocessor
from lc_classifier.features.composites.ztf import ZTFFeatureExtractor
from joblib import Parallel, delayed
from dataset import save_batch

folder = 'data_231130'
astro_objects = pd.read_pickle(os.path.join(folder, 'astro_objects_without_features.pkl'))

batch_size = 100
n_batches = int(np.ceil(len(astro_objects) / batch_size))


def extract_features(batch_id, batch_astro_objects, shorten_n_days=None):
    lightcurve_preprocessor = ZTFLightcurvePreprocessor()
    lightcurve_preprocessor.preprocess_batch(batch_astro_objects)
    if shorten_n_days is not None:
        shorten_preprocessor = ShortenPreprocessor(shorten_n_days)
        shorten_preprocessor.preprocess_batch(batch_astro_objects)

    feature_extractor = ZTFFeatureExtractor()
    feature_extractor.compute_features_batch(batch_astro_objects, progress_bar=False)

    filename = os.path.join(
        folder,
        f'astro_objects_batch_{shorten_n_days}_{batch_id:04}.pkl')
    save_batch(batch_astro_objects, filename)


# n_days = [16, 32, 64, 128, 256, None]
n_days = [None]
for shorten_n_days in n_days:
    tasks = []
    for batch_id in range(n_batches):
        batch_astro_objects = astro_objects[batch_id*batch_size:(batch_id+1)*batch_size]
        tasks.append(
            delayed(extract_features)(
                batch_id,
                batch_astro_objects,
                shorten_n_days
            )
        )

    Parallel(n_jobs=14, verbose=11, backend="loky")(tasks)
