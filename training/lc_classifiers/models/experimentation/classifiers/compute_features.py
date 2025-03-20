import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from joblib import Parallel, delayed

folder = "data_231206"
output_folder = folder + "_features"
if not os.path.exists(output_folder):
    os.mkdir(output_folder)

astro_objects_filenames = os.listdir(folder)
astro_objects_filenames = [
    f for f in astro_objects_filenames if "astro_objects_batch" in f
]


def extract_features(
    batch_id, ao_filename, shorten_n_days=None, skip_if_output_exists=False
):
    import pandas as pd
    from lc_classifier.features.preprocess.ztf import (
        ZTFLightcurvePreprocessor,
        ShortenPreprocessor,
    )
    from lc_classifier.features.composites.ztf import ZTFFeatureExtractor
    from lc_classifier.features.core.base import astro_object_from_dict
    from lc_classifier.features.core.base import save_batch

    output_filename = os.path.join(
        output_folder, f"astro_objects_batch_{shorten_n_days}_{batch_id:04}.pkl"
    )

    if skip_if_output_exists and os.path.exists(output_filename):
        return

    batch_astro_objects = pd.read_pickle(os.path.join(folder, ao_filename))
    batch_astro_objects = [astro_object_from_dict(d) for d in batch_astro_objects]

    lightcurve_preprocessor = ZTFLightcurvePreprocessor()
    lightcurve_preprocessor.preprocess_batch(batch_astro_objects)
    if shorten_n_days is not None:
        shorten_preprocessor = ShortenPreprocessor(shorten_n_days)
        shorten_preprocessor.preprocess_batch(batch_astro_objects)

    feature_extractor = ZTFFeatureExtractor()
    feature_extractor.compute_features_batch(batch_astro_objects, progress_bar=False)

    save_batch(batch_astro_objects, output_filename)


def patch_features(batch_id, shorten_n_days=None):
    import pandas as pd
    from typing import List

    from lc_classifier.features.core.base import (
        FeatureExtractorComposite,
        FeatureExtractor,
    )
    from lc_classifier.features.extractors.tde_extractor import TDETailExtractor
    from lc_classifier.features.core.base import astro_object_from_dict
    from lc_classifier.features.core.base import save_batch

    filename = os.path.join(
        output_folder, f"astro_objects_batch_{shorten_n_days}_{batch_id:04}.pkl"
    )

    batch_astro_objects = pd.read_pickle(filename)
    batch_astro_objects = [astro_object_from_dict(d) for d in batch_astro_objects]
    # Assuming light curve is preprocessed and shortened

    # Delete old features to be patched
    features_to_be_patched = ["TDE_decay", "TDE_decay_chi"]

    for ao in batch_astro_objects:
        features = ao.features
        features = features[~features["name"].isin(features_to_be_patched)]
        ao.features = features

    class PatchExtractor(FeatureExtractorComposite):
        version = "1.0.0"

        def _instantiate_extractors(self) -> List[FeatureExtractor]:
            bands = list("gr")

            feature_extractors = [
                TDETailExtractor(bands),
            ]
            return feature_extractors

    feature_extractor = PatchExtractor()
    feature_extractor.compute_features_batch(batch_astro_objects, progress_bar=False)

    save_batch(batch_astro_objects, filename)


n_days = [16, 32, 64, 128, 256, 512, 1024, None]
for shorten_n_days in n_days:
    tasks = []
    for ao_filename in astro_objects_filenames:
        batch_id = int(ao_filename.split(".")[0].split("_")[3])
        tasks.append(delayed(patch_features)(batch_id, shorten_n_days))

    Parallel(n_jobs=9, verbose=11, backend="loky")(tasks)

exit()

n_days = [16, 32, 64, 128, 256, 512, 1024, None]
for shorten_n_days in n_days:
    tasks = []
    for ao_filename in astro_objects_filenames:
        batch_id = int(ao_filename.split(".")[0].split("_")[3])
        tasks.append(delayed(extract_features)(batch_id, ao_filename, shorten_n_days))

    Parallel(n_jobs=12, verbose=11, backend="loky")(tasks)
