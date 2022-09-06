from .dict_transform import FEAT_DICT

import numpy as np
import pandas as pd


class ELAsTiCCMapper:
    _fid_mapper = {
        0: "u",
        1: "g",
        2: "r",
        3: "i",
        4: "z",
        5: "Y",
    }
    _rename_cols = {
        "mag": "FLUXCAL",
        "e_mag": "FLUXCALERR",
        "fid": "BAND",
        "mjd": "MJD",
    }
    feat_dict = FEAT_DICT

    @classmethod
    def get_detections(cls, light_curves: pd.DataFrame) -> pd.DataFrame:
        # Get detections
        exploded_det = light_curves.explode("detections")
        detections = pd.DataFrame.from_records(
            exploded_det["detections"].values, index=exploded_det.index
        )
        detections["type"] = "d"
        # Get forced photometry and remove NaN
        expl_forced_phot = light_curves.explode("non_detections")
        expl_forced_phot = expl_forced_phot[~expl_forced_phot["non_detections"].isna()]
        # Get the usable keys
        cols = list(cls._rename_cols.keys()) + ["type"]

        if not expl_forced_phot.empty:
            forced_phot = pd.DataFrame.from_records(
                expl_forced_phot["non_detections"].values, index=expl_forced_phot.index
            )
            forced_phot.rename(columns={"diffmaglim": "mag"}, inplace=True)
            forced_phot["e_mag"] = forced_phot["extra_fields"].apply(
                lambda x: x["psFluxErr"]
            )
            forced_phot["type"] = "f"
        else:
            forced_phot = pd.DataFrame(columns=cols)

        detections = detections[cols]
        forced_phot = forced_phot[cols]

        # Concat detections with forced photometry
        detections = pd.concat([detections, forced_phot])
        detections = detections.rename(columns=cls._rename_cols)
        detections["BAND"] = detections["BAND"].map(lambda x: cls._fid_mapper[x])
        detections.index.name = "aid"  # force to put this name
        return detections

    @classmethod
    def get_header(cls, light_curves: pd.DataFrame, keep="first") -> pd.DataFrame:
        exploded = light_curves.explode("detections")
        detections = pd.DataFrame.from_records(
            exploded["detections"].values, index=exploded.index
        )
        detections = detections.sort_values(by=["mjd"])
        headers = pd.DataFrame.from_records(
            detections["extra_fields"].values, index=detections.index
        )
        headers = headers[headers["diaObject"].notnull()]
        headers = headers[~headers.index.duplicated(keep=keep)]
        headers = pd.DataFrame.from_records(
            headers["diaObject"].values, index=headers.index
        )
        headers = headers[list(cls.feat_dict.keys())]
        headers = headers.rename(columns=cls.feat_dict)
        headers = headers.sort_index()
        return headers

    @classmethod
    def get_features(cls, light_curve_and_features: pd.DataFrame) -> pd.DataFrame:
        features = light_curve_and_features[
            ~light_curve_and_features["features"].isna()
        ]
        features = pd.DataFrame.from_records(
            features["features"].values, index=features.index
        )
        features.replace({None: np.nan}, inplace=True)
        return features
