from ..core.base import FeatureExtractor, AstroObject
from typing import List
import numpy as np
import pandas as pd
from functools import lru_cache


class PanStarrsFeatureExtractor(FeatureExtractor):
    version = "1.0.1"

    def __init__(self):
        self.required_metadata = [
            "sgscore1",
            "distpsnr1",
            "sgmag1",
            "srmag1",
            "simag1",
            "szmag1",
        ]

    def compute_features_single_object(self, astro_object: AstroObject):
        metadata = astro_object.metadata
        features = []

        available_fields = set(metadata["name"].values)
        field_intersection = available_fields.intersection(set(self.required_metadata))
        if len(field_intersection) != len(self.required_metadata):
            features.append(["sgscore1", np.nan])
            features.append(["distpsnr1", np.nan])
            features.append(["ps_g-r", np.nan])
            features.append(["ps_r-i", np.nan])
            features.append(["ps_i-z", np.nan])
        else:
            sg_score = metadata[metadata["name"] == "sgscore1"]["value"].values[0]
            dist_nr = metadata[metadata["name"] == "distpsnr1"]["value"].values[0]

            if sg_score < 0 or dist_nr < 0:
                sg_score = np.nan
                dist_nr = np.nan

            features.append(["sgscore1", sg_score])
            features.append(["distpsnr1", dist_nr])

            g_mag = metadata[metadata["name"] == "sgmag1"]["value"].values[0]
            r_mag = metadata[metadata["name"] == "srmag1"]["value"].values[0]
            i_mag = metadata[metadata["name"] == "simag1"]["value"].values[0]
            z_mag = metadata[metadata["name"] == "szmag1"]["value"].values[0]

            if g_mag < -30.0 or r_mag < -30.0:
                color_gr = np.nan
            else:
                color_gr = g_mag - r_mag

            features.append(["ps_g-r", color_gr])

            if r_mag < -30.0 or i_mag < -30.0:
                color_ri = np.nan
            else:
                color_ri = r_mag - i_mag

            features.append(["ps_r-i", color_ri])

            if i_mag < -30.0 or z_mag < -30.0:
                color_iz = np.nan
            else:
                color_iz = i_mag - z_mag

            features.append(["ps_i-z", color_iz])

        features_df = pd.DataFrame(data=features, columns=["name", "value"])

        features_df["fid"] = None
        features_df["sid"] = "panstarrs"
        features_df["version"] = self.version

        all_features = [astro_object.features, features_df]
        astro_object.features = pd.concat(
            [f for f in all_features if not f.empty], axis=0
        )
