from ..core.base import FeatureExtractor, AstroObject
import numpy as np
import pandas as pd
from typing import List
import mhps


class MHPSExtractor(FeatureExtractor):
    def __init__(
        self,
        bands: List[str],
        unit: str,
        t1: float = 100.0,
        t2: float = 10.0,
        dt: float = 3.0,
    ):
        self.version = "1.0.0"

        self.t1 = t1
        self.t2 = t2
        self.dt = dt

        self.bands = bands
        valid_units = ["magnitude", "diff_flux"]
        if unit not in valid_units:
            raise ValueError(f"{unit} is not a valid unit ({valid_units})")
        self.unit = unit

    def preprocess_detections(self, detections: pd.DataFrame) -> pd.DataFrame:
        detections = detections[detections["unit"] == self.unit]
        detections = detections[detections["brightness"].notna()]
        return detections

    def compute_features_single_object(self, astro_object: AstroObject):
        detections = astro_object.detections
        detections = self.preprocess_detections(detections)

        features = []
        for band in self.bands:
            band_detections = detections[detections["fid"] == band].copy()
            if len(band_detections) == 0:
                ratio = low = high = non_zero = pn_flag = np.nan
            else:
                band_detections.sort_values("mjd", inplace=True)

                time = band_detections["mjd"].values
                brightness = band_detections["brightness"].values
                e_brightness = band_detections["e_brightness"].values

                if self.unit == "magnitude":
                    brightness = brightness.astype(np.double)
                    e_brightness = e_brightness.astype(np.double)
                    time = time.astype(np.double)

                    # TODO: check extra params of mhps.statistics
                    ratio, low, high, non_zero, pn_flag = mhps.statistics(
                        brightness, e_brightness, time, self.t1, self.t2
                    )
                elif self.unit == "diff_flux":
                    brightness = brightness.astype(np.float32)
                    e_brightness = e_brightness.astype(np.float32)
                    time = time.astype(np.float32)

                    ratio, low, high, non_zero, pn_flag = mhps.flux_statistics(
                        brightness, e_brightness, time, self.t1, self.t2
                    )

            if self.t1 == 100.0 and self.t2 == 10.0:
                features.append((f"MHPS_ratio", ratio, band))
                features.append((f"MHPS_low", low, band))
                features.append((f"MHPS_high", high, band))
                features.append((f"MHPS_non_zero", non_zero, band))
                features.append((f"MHPS_PN_flag", pn_flag, band))
            else:
                features.append(
                    (f"MHPS_ratio_{int(self.t1)}_{int(self.t2)}", ratio, band)
                )
                features.append((f"MHPS_low_{int(self.t1)}", low, band))
                features.append((f"MHPS_high_{int(self.t2)}", high, band))

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
