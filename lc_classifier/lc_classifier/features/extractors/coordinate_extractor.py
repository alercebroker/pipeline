import pandas as pd

from ..core.base import FeatureExtractor, AstroObject
import numpy as np


class CoordinateExtractor(FeatureExtractor):
    version = "1.0.0"

    def compute_features_single_object(self, astro_object: AstroObject):
        detections = astro_object.detections

        # "conversion -> mean" would be better, but this is cheaper
        ra_deg, dec_deg = detections[["ra", "dec"]].mean().values

        """ Right ascension and declination to cartesian coordinates in a unit sphere"""
        ra_rad = ra_deg / 180.0 * np.pi
        dec_rad = dec_deg / 180.0 * np.pi
        x = np.cos(ra_rad) * np.cos(dec_rad)
        y = np.sin(ra_rad) * np.cos(dec_rad)
        z = np.sin(dec_rad)

        features = [("Coordinate_x", x), ("Coordinate_y", y), ("Coordinate_z", z)]

        features_df = pd.DataFrame(data=features, columns=["name", "value"])
        features_df["fid"] = None
        sids = astro_object.detections["sid"].unique()
        sids = np.sort(sids)
        sid = ",".join(sids)

        features_df["sid"] = sid
        features_df["version"] = self.version
        all_features = [astro_object.features, features_df]
        astro_object.features = pd.concat(
            [f for f in all_features if not f.empty], axis=0
        )
