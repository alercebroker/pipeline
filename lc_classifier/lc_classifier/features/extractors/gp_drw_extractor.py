from ..core.base import FeatureExtractor, AstroObject
from typing import List, Dict
import numpy as np
import pandas as pd
import celerite2
from celerite2 import terms
from scipy.optimize import minimize


class GPDRWExtractor(FeatureExtractor):
    def __init__(self, bands: List[str], unit: str):
        self.version = "1.0.0"
        self.bands = bands
        valid_units = ["magnitude", "diff_flux"]
        if unit not in valid_units:
            raise ValueError(f"{unit} is not a valid unit ({valid_units})")
        self.unit = unit
        self.detections_min_len = 5

    def preprocess_detections(self, detections: pd.DataFrame) -> pd.DataFrame:
        detections = detections[detections["brightness"].notna()]
        detections = detections[detections["unit"] == self.unit]

        return detections

    def compute_features_single_object(self, astro_object: AstroObject):
        detections = astro_object.detections
        detections = self.preprocess_detections(detections)

        features = []
        for band in self.bands:
            detections_band = detections[detections["fid"] == band].copy()
            band_features = {
                "GP_DRW_sigma": np.nan,
                "GP_DRW_tau": np.nan,
            }
            if len(detections_band) < self.detections_min_len:
                features += self._dict_to_features(band_features, band)
                continue

            detections_band["mjd"] -= np.min(detections_band["mjd"])
            detections_band.sort_values("mjd", inplace=True)

            detections_band["brightness"] -= np.mean(detections_band["brightness"])

            kernel = terms.RealTerm(a=1.0, c=10.0)
            gp = celerite2.GaussianProcess(kernel, mean=0.0)

            def neg_log_like(params, gp, time, mag, sq_error):
                gp.mean = 0.0
                theta = np.exp(params)
                gp.kernel = terms.RealTerm(a=theta[0], c=theta[1])
                gp.compute(time, diag=sq_error, quiet=True)
                return -gp.log_likelihood(mag)

            initial_params = np.zeros((2,), dtype=float)
            sol = minimize(
                neg_log_like,
                initial_params,
                bounds=[[-10.0, 19.0], [-6.0, 6.0]],
                method="L-BFGS-B",
                args=(
                    gp,
                    detections_band["mjd"].values,
                    detections_band["brightness"].values,
                    detections_band["e_brightness"].values ** 2,
                ),
                # options={'iprint': 99}
            )

            optimal_params = np.exp(sol.x)

            band_features["GP_DRW_sigma"] = optimal_params[0]
            band_features["GP_DRW_tau"] = 1.0 / optimal_params[1]

            features += self._dict_to_features(band_features, band)

        features_df = pd.DataFrame(data=features, columns=["name", "value", "fid"])

        sids = detections["sid"].unique()
        sids = np.sort(sids)
        sid = ",".join(sids)

        features_df["sid"] = sid
        features_df["version"] = self.version

        all_features = [astro_object.features, features_df]
        astro_object.features = pd.concat(
            [f for f in all_features if not f.empty], axis=0
        )

    def _dict_to_features(self, band_features: Dict[str, float], band: str) -> List:
        features = []
        for k, v in band_features.items():
            features.append((k, v, band))

        return features
