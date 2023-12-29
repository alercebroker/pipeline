from ..core.base import FeatureExtractor
from lc_classifier.base import AstroObject
import numpy as np
import pandas as pd
from typing import List


class TDEExtractor(FeatureExtractor):
    version = '1.0.0'
    unit = 'magnitude'

    def __init__(self, bands: List[str]):
        self.bands = bands

    def get_observations(self, astro_object: AstroObject) -> pd.DataFrame:
        observations = astro_object.detections
        if astro_object.forced_photometry is not None:
            observations = pd.concat([
                observations,
                astro_object.forced_photometry], axis=0)
        observations = observations[observations['unit'] == self.unit]
        observations = observations[observations['brightness'].notna()]
        observations = observations[observations['e_brightness'] > 0.0]
        observations = observations[observations['e_brightness'] < 1.0]
        return observations

    def compute_features_single_object(self, astro_object: AstroObject):
        observations = pd.concat([
            astro_object.detections,
            astro_object.forced_photometry],
            axis=0
        )
        observations = observations[observations['unit'] == 'magnitude']
        features = []
        for band in self.bands:
            band_observations = observations[observations['fid'] == band]
            if len(band_observations) < 2:
                features.append(('TDE_decay', np.nan, band))
                features.append(('TDE_decay_chi', np.nan, band))
                continue

            brightest_obs = band_observations.sort_values('brightness').iloc[0]
            t_d = brightest_obs.mjd + 14

            after_t_d = band_observations[band_observations['mjd'] > t_d]

            x = 2.5 * np.log10(after_t_d.mjd.values - t_d + 30)
            y = after_t_d.brightness.values
            y_err = after_t_d.e_brightness.values + 1e-2

            omega = np.stack([
                np.ones(len(x)),
                x
            ], axis=-1)
            inverr = 1.0 / y_err

            # weighted regularized linear regression
            w_a = inverr.reshape(-1, 1) * omega
            w_b = (y * inverr).reshape(-1, 1)
            coeffs = np.matmul(np.linalg.pinv(w_a), w_b).flatten()

            # Calculate reduced chi-squared statistic
            fitted_magnitude = coeffs[1] * x + coeffs[0]
            chi = np.sum((fitted_magnitude - y) ** 2 / y_err ** 2)
            chi_den = len(fitted_magnitude) - 2
            if chi_den >= 1:
                chi_per_degree = chi / chi_den
            else:
                chi_per_degree = np.nan

            features.append(('TDE_decay', coeffs[1], band))
            features.append(('TDE_decay_chi', chi_per_degree, band))

        features_df = pd.DataFrame(
            data=features,
            columns=['name', 'value', 'fid']
        )

        sids = astro_object.detections['sid'].unique()
        sids = np.sort(sids)
        sid = ','.join(sids)

        features_df['sid'] = sid
        features_df['version'] = self.version

        all_features = [astro_object.features, features_df]
        astro_object.features = pd.concat(
            [f for f in all_features if not f.empty],
            axis=0
        )
