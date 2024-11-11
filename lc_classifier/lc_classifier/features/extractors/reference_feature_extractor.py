# Based on the work of Manuel Pavez

from ..core.base import FeatureExtractor, AstroObject
import numpy as np
import pandas as pd

from typing import List


class ReferenceFeatureExtractor(FeatureExtractor):
    version = "1.0.0"
    unit = "diff_flux"

    def __init__(self, bands: List[str]):
        self.bands = bands

    def get_observations(self, astro_object: AstroObject) -> pd.DataFrame:
        observations = astro_object.detections
        if astro_object.forced_photometry is not None:
            observations = pd.concat(
                [observations, astro_object.forced_photometry], axis=0
            )
        observations = observations[observations["unit"] == self.unit]
        observations = observations[observations["brightness"].notna()]
        observations = observations[observations["e_brightness"] > 0.0]
        observations = observations[observations["fid"].isin(self.bands)]

        # 5 arcsec limit is because of training set limitations (it used the ZTF forced photometry service)
        mask = (observations["distnr"] != -99999) & (observations["distnr"] <= 5.0)
        observations = observations[mask].copy()
        return observations

    def compute_features_single_object(self, astro_object: AstroObject):
        observations = self.get_observations(astro_object)

        reference = pd.DataFrame()
        if astro_object.reference is not None:
            reference = astro_object.reference.copy()

        features = []
        all_bands = ",".join(self.bands)

        distnr = observations["distnr"][observations["distnr"].notna()]
        if len(distnr) == 0:
            features.append(["mean_distnr", np.nan, all_bands])
            features.append(["sigma_distnr", np.nan, all_bands])
        else:
            mean_distnr = observations["distnr"].mean()
            sigma_distnr = observations["distnr"].std()

            features.append(["mean_distnr", mean_distnr, all_bands])
            features.append(["sigma_distnr", sigma_distnr, all_bands])

        if len(observations) == 0 or len(reference) == 0:
            features.append(["mean_sharpnr", np.nan, all_bands])
            features.append(["mean_chinr", np.nan, all_bands])
        else:
            mask = observations["rfid"].notna()
            observations = observations.loc[mask, ["distnr", "rfid"]].copy()
            observations.set_index("rfid", inplace=True)

            reference.set_index("rfid", inplace=True)
            ref_index = observations.index.intersection(reference.index)
            columns = ["sharpnr", "chinr"]
            observations.loc[ref_index, columns] = reference.loc[ref_index, columns]

            if len(observations.loc[ref_index]) == 0:
                features.append(["mean_sharpnr", np.nan, all_bands])
                features.append(["mean_chinr", np.nan, all_bands])
            else:
                mean_sharpnr = observations["sharpnr"].mean()
                mean_chinr = observations["chinr"].mean()
                features.append(["mean_sharpnr", mean_sharpnr, all_bands])
                features.append(["mean_chinr", mean_chinr, all_bands])

        features_df = pd.DataFrame(data=features, columns=["name", "value", "fid"])

        sids = astro_object.detections["sid"].unique()
        sids = np.sort(sids)
        sid = ",".join(sids)

        features_df["sid"] = sid
        features_df["version"] = self.version

        all_features = [astro_object.features, features_df]
        astro_object.features = pd.concat(
            [f for f in all_features if not f.empty], axis=0
        )
