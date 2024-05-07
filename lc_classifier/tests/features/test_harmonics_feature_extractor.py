import unittest
import numpy as np
import matplotlib.pyplot as plt
from lc_classifier.features.extractors.harmonics_extractor import HarmonicsExtractor
from lc_classifier.examples.data import get_ztf_example


class TestHarmonicsFeatureExtractor(unittest.TestCase):
    def test_ztf_object(self):
        ztf_astro_object = get_ztf_example(1)
        period_value = 0.5388739365
        ztf_astro_object.features.loc[0] = [
            "Multiband_period",
            period_value,
            "g,r",
            "ztf_survey",
            "1.0.0",
        ]
        feature_extractor = HarmonicsExtractor(
            bands=["g", "r"], unit="magnitude", use_forced_photo=True
        )
        feature_extractor.compute_features_single_object(ztf_astro_object)

        observations = feature_extractor.get_observations(ztf_astro_object)
        r_obs = observations[observations["fid"] == "r"]
        plot = False
        if plot:
            plt.scatter(
                r_obs["mjd"] % period_value,
                r_obs["brightness"] - np.mean(r_obs["brightness"]),
                c="b",
            )
            plt.scatter(
                r_obs["mjd"] % period_value + period_value,
                r_obs["brightness"] - np.mean(r_obs["brightness"]),
                c="b",
            )
            time_grid = np.linspace(0, period_value, 1000)
            r_harmonics = ztf_astro_object.features[
                ztf_astro_object.features["fid"] == "r"
            ].copy()
            r_harmonics.set_index("name", inplace=True)
            r_harmonics = r_harmonics["value"]
            offset = 0.32
            brightness = r_harmonics.loc["Harmonics_mag_1"] * np.sin(
                2 * np.pi / period_value * (time_grid + offset)
            )
            for i in range(2, 8):
                phase = r_harmonics.loc[f"Harmonics_phase_{i}"]
                brightness += r_harmonics.loc[f"Harmonics_mag_{i}"] * np.sin(
                    i * 2 * np.pi / period_value * (time_grid + offset) + phase
                )

            plt.plot(time_grid, brightness, "orange")
            plt.plot(time_grid + period_value, brightness, "orange")
            plt.show()

        r_harmonics = ztf_astro_object.features[
            ztf_astro_object.features["fid"] == "r"
        ].copy()
        r_harmonics.set_index("name", inplace=True)
        r_harmonics = r_harmonics["value"]
        offset = 0.32  # eye-calibrated
        time = r_obs["mjd"].values
        brightness = r_harmonics.loc["Harmonics_mag_1"] * np.sin(
            2 * np.pi / period_value * (time + offset)
        )
        for i in range(2, 8):
            phase = r_harmonics.loc[f"Harmonics_phase_{i}"]
            brightness += r_harmonics.loc[f"Harmonics_mag_{i}"] * np.sin(
                i * 2 * np.pi / period_value * (time + offset) + phase
            )

        measured_brightness = r_obs["brightness"] - np.mean(r_obs["brightness"])
        average_abs_error = np.mean(np.abs(brightness - measured_brightness))
        self.assertLessEqual(average_abs_error, 0.3)
