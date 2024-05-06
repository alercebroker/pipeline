from ..core.base import FeatureExtractor, AstroObject
from typing import List
import numpy as np
import pandas as pd
from functools import lru_cache


class PanStarrsFeatureExtractor(FeatureExtractor):
    version = "1.0.0"

    def __init__(self):
        self.required_metadata = ["sgscore1", "distpsnr1", "sgmag1", "srmag1"]

    def compute_features_single_object(self, astro_object: AstroObject):
        metadata = astro_object.metadata
        features = []

        available_fields = set(metadata["name"].values)
        field_intersection = available_fields.intersection(set(self.required_metadata))
        if len(field_intersection) != len(self.required_metadata):
            features.append(["sg_score", np.nan])
            features.append(["dist_nr", np.nan])
            features.append(["ps_g-r", np.nan])
        else:
            sg_score = metadata[metadata["name"] == "sgscore1"]["value"].values[0]
            dist_nr = metadata[metadata["name"] == "distpsnr1"]["value"].values[0]

            if sg_score < 0 or dist_nr < 0:
                sg_score = np.nan
                dist_nr = np.nan

            features.append(["sg_score", sg_score])
            features.append(["dist_nr", dist_nr])

            g_mag = metadata[metadata["name"] == "sgmag1"]["value"].values[0]
            r_mag = metadata[metadata["name"] == "srmag1"]["value"].values[0]

            if g_mag < -30.0 or r_mag < -30.0:
                color = np.nan
            else:
                color = g_mag - r_mag

            features.append(["ps_g-r", color])

        features_df = pd.DataFrame(data=features, columns=["name", "value"])

        features_df["fid"] = None
        features_df["sid"] = "panstarrs"
        features_df["version"] = self.version

        all_features = [astro_object.features, features_df]
        astro_object.features = pd.concat(
            [f for f in all_features if not f.empty], axis=0
        )
