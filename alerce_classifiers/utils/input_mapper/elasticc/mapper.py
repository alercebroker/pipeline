from .dict_transform import FEAT_DICT

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
        exploded = light_curves.explode("detections")
        detections = pd.DataFrame.from_records(
            exploded["detections"].values, index=exploded.index
        )
        detections = detections[cls._rename_cols.keys()]
        detections = detections.rename(columns=cls._rename_cols)
        detections["BAND"] = detections["BAND"].map(lambda x: cls._fid_mapper[x])
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
