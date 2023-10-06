import pandas as pd

from ..core.base import FeatureExtractor, AstroObject
import numpy as np
from typing import List


class ColorFeatureExtractor(FeatureExtractor):
    def __init__(self, bands: List[str], unit: str):
        self.version = '1.0.0'
        self.bands = bands
        valid_units = ['magnitude']
        if unit not in valid_units:
            raise ValueError(f'{unit} is not a valid unit ({valid_units})')
        self.unit = unit

    def preprocess_detections(self, detections: pd.DataFrame) -> pd.DataFrame:
        detections = detections[detections['brightness'].notna()]

        if self.unit == 'magnitude':
            detections = detections[detections['e_brightness'] < 1.0]

        return detections

    def compute_features_single_object(self, astro_object: AstroObject):
        detections = astro_object.detections
        detections = self.preprocess_detections(detections)
        sids = detections[detections['fid'].isin(self.bands)]['sid'].unique()
        fid = ','.join(self.bands)
        sid = ','.join(sids)

        # compute mean and max of each band
        band_means = []
        band_maxima = []
        for band in self.bands:
            band_detections = detections[detections['fid'] == band]
            if len(band_detections) == 0:
                band_means.append(np.nan)
                band_maxima.append(np.nan)
            else:
                band_mean = np.mean(band_detections['brightness'])
                band_means.append(band_mean)
                band_max = np.max(band_detections['brightness'])
                band_maxima.append(band_max)

        features = []
        for i in range(len(self.bands)-1):
            feature_name = f'{self.bands[i]}-{self.bands[i+1]}_mean'
            feature_value = band_means[i] - band_means[i+1]
            features.append([
                feature_name,
                feature_value,
                fid,
                sid,
                self.version
            ])

            feature_name = f'{self.bands[i]}-{self.bands[i+1]}_max'
            feature_value = band_maxima[i] - band_maxima[i+1]
            features.append([
                feature_name,
                feature_value,
                fid,
                sid,
                self.version
            ])

        features_df = pd.DataFrame(
            data=features,
            columns=astro_object.features.columns
        )

        astro_object.features = pd.concat(
            [astro_object.features, features_df],
            axis=0
        )
