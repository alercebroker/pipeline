import unittest
from lc_classifier.examples.data import get_ztf_forced_training_examples
from lc_classifier.features.composites.ztf import ZTFFeatureExtractor
import time


class TestZTFExtractor(unittest.TestCase):
    def test_ztf_forced(self):
        astro_objects = get_ztf_forced_training_examples()
        feature_extractor = ZTFFeatureExtractor()
        feature_extractor.compute_features_batch(astro_objects, progress_bar=False)

        astro_objects = get_ztf_forced_training_examples()
        t0 = time.time()
        feature_extractor.compute_features_batch(astro_objects, progress_bar=False)
        dt = time.time() - t0
        average_dt = dt / len(astro_objects)
        print(average_dt, "[s] per object on avg.")
        self.assertLessEqual(average_dt, 3.0)

        # all feature df must be the same shape
        feature_shapes = [astro_object.features.shape for astro_object in astro_objects]
        feature_shapes = set(feature_shapes)
        self.assertEqual(len(feature_shapes), 1)
        print(astro_objects[0].features["name"].unique())


if __name__ == "__main__":
    astro_objects = get_ztf_forced_training_examples()
    feature_extractor = ZTFFeatureExtractor()
    feature_extractor.compute_features_batch(astro_objects, progress_bar=False)

    import cProfile

    astro_objects = get_ztf_forced_training_examples()
    profiler = cProfile.Profile()
    profiler.enable()
    feature_extractor.compute_features_batch(astro_objects, progress_bar=False)
    profiler.disable()
    profiler.dump_stats("profiler_stats.prof")
