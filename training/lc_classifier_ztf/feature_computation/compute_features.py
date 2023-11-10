import numpy as np
import pandas as pd
from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor
from lc_classifier.features.composites.ztf import ZTFFeatureExtractor
from joblib import Parallel, delayed
from dataset import save_batch


astro_objects = pd.read_pickle('data/astro_objects_without_features.pkl')

batch_size = 100
n_batches = int(np.ceil(len(astro_objects) / batch_size))


def extract_features(batch_id, batch_astro_objects):
    lightcurve_preprocessor = ZTFLightcurvePreprocessor()
    lightcurve_preprocessor.preprocess_batch(batch_astro_objects)

    feature_extractor = ZTFFeatureExtractor()
    feature_extractor.compute_features_batch(batch_astro_objects, progress_bar=False)
    save_batch(batch_astro_objects, batch_id)


tasks = []
for batch_id in range(n_batches):
    batch_astro_objects = astro_objects[batch_id*batch_size:(batch_id+1)*batch_size]
    tasks.append(
        delayed(extract_features)(
            batch_id,
            batch_astro_objects)
    )

Parallel(n_jobs=10, verbose=11, backend="loky")(tasks)
