import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from joblib import Parallel, delayed

folder = 'data_231206'
output_folder = folder+'_features'
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

astro_objects_filenames = os.listdir(folder)
astro_objects_filenames = [f for f in astro_objects_filenames if 'astro_objects_batch' in f]


def extract_features(batch_id, ao_filename, shorten_n_days=None):
    import pandas as pd
    from lc_classifier.features.preprocess.ztf import ZTFLightcurvePreprocessor, ShortenPreprocessor
    from lc_classifier.features.composites.ztf import ZTFFeatureExtractor
    from dataset import save_batch

    batch_astro_objects = pd.read_pickle(
        os.path.join(folder, ao_filename))

    lightcurve_preprocessor = ZTFLightcurvePreprocessor()
    lightcurve_preprocessor.preprocess_batch(batch_astro_objects)
    if shorten_n_days is not None:
        shorten_preprocessor = ShortenPreprocessor(shorten_n_days)
        shorten_preprocessor.preprocess_batch(batch_astro_objects)

    feature_extractor = ZTFFeatureExtractor()
    feature_extractor.compute_features_batch(batch_astro_objects, progress_bar=False)

    filename = os.path.join(
        output_folder,
        f'astro_objects_batch_{shorten_n_days}_{batch_id:04}.pkl')
    save_batch(batch_astro_objects, filename)


# n_days = [16, 32, 64, 128, 256, None]
n_days = [None]
for shorten_n_days in n_days:
    tasks = []
    for ao_filename in astro_objects_filenames:
        batch_id = int(ao_filename.split('.')[0].split('_')[3])
        tasks.append(
            delayed(extract_features)(
                batch_id,
                ao_filename,
                shorten_n_days
            )
        )

    Parallel(n_jobs=6, verbose=11, backend="loky")(tasks)
