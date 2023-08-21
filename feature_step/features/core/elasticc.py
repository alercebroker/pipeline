from lc_classifier.features.custom.elasticc_feature_extractor import (
    ElasticcFeatureExtractor,
)
from lc_classifier.features.preprocess.preprocess_elasticc import (
    ElasticcPreprocessor,
)
from typing import List
import pandas as pd
import pickle
import os


class ELAsTiCCFeatureExtractor:
    NAME = "elasticc_lc_features"
    VERSION = os.getenv("EXTRACTOR_VERSION", "0.0.0")
    SURVEYS = ("LSST",)
    BANDS = ("u", "g", "r", "i", "z", "Y")
    BANDS_MAPPING = {}

    def __init__(
        self,
        detections: List[dict],
        non_detections: List[dict],
        xmatch: List[dict],
        **kwargs,
    ):
        self.preprocessor = kwargs.get("preprocessor", ElasticcPreprocessor())
        self.extractor = kwargs.get(
            "extractor", ElasticcFeatureExtractor(round=2)
        )
        self.detections = detections

    def generate_features(self):
        lightcurves = self._create_lightcurve_dataframe(self.detections)
        metadata = self._create_metadata_dataframe(self.detections)
        input_snids = lightcurves.index.unique().values

        lightcurves = self.preprocessor.preprocess(lightcurves)
        features = self.extractor.compute_features(
            lightcurves, metadata=metadata, force_snids=input_snids
        )
        return features

    def _create_lightcurve_dataframe(self, detections: List[dict]):
        self._rename_detections_columns(detections)
        return pd.DataFrame.from_records(detections).set_index("aid")

    def _rename_detections_columns(self, detections: List[dict]):
        for det in detections:
            det["MJD"] = det.pop("mjd")
            det["FLUXCAL"] = det.pop("mag")
            det["FLUXCALERR"] = det.pop("e_mag")
            det["BAND"] = det.pop("fid")

    def _create_metadata_dataframe(self, detections: List[dict]):
        metadata = []
        aids = []
        for det in detections:
            det["extra_fields"]["diaObject"] = pickle.loads(
                det["extra_fields"]["diaObject"]
            )[0]
            det["extra_fields"]["diaObject"] = self._map_metadata_column_names(
                det["extra_fields"]["diaObject"]
            )
            metadata.append(det["extra_fields"]["diaObject"])
            aids.append(det["aid"])

        dataframe = pd.DataFrame.from_records(metadata)
        dataframe["aid"] = aids
        dataframe.drop_duplicates("aid", inplace=True)
        dataframe.set_index("aid", inplace=True)
        return dataframe

    def _map_metadata_column_names(self, diaObject: dict):
        mapping = {
            "z_final": "REDSHIFT_HELIO",
            "z_final_err": "REDSHIFT_HELIO_ERR",
            "hostgal_zphot": "HOSTGAL_PHOTOZ",
            "hostgal_zphot_err": "HOSTGAL_PHOTOZ_ERR",
            "hostgal2_zphot": "HOSTGAL2_PHOTOZ",
            "hostgal2_zphot_err": "HOSTGAL2_PHOTOZ_ERR",
            "hostgal2_zspec": "HOSTGAL2_SPECZ",
            "hostgal2_zspec_err": "HOSTGAL2_SPECZ_ERR",
            "hostgal_zspec": "HOSTGAL_SPECZ",
            "hostgal_zspec_err": "HOSTGAL_SPECZ_ERR",
        }
        for key in diaObject.copy():
            if key in mapping:
                diaObject[mapping[key]] = diaObject.pop(key)
                continue
            diaObject[key.upper()] = diaObject.pop(key)
        return diaObject
