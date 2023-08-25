from lc_classifier.features.custom.elasticc_feature_extractor import (
    ElasticcFeatureExtractor,
)
from lc_classifier.features.preprocess.preprocess_elasticc import (
    ElasticcPreprocessor,
)
from typing import List
import pandas as pd
import copy
from importlib import metadata


class ELAsTiCCFeatureExtractor:
    NAME = "elasticc_lc_features"
    VERSION = metadata.version("feature-step")
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
        self.preprocessor = kwargs.get(
            "preprocessor", ElasticcPreprocessor(stream=True)
        )
        self.extractor = kwargs.get(
            "extractor", ElasticcFeatureExtractor(round=2)
        )
        self.detections = detections

    def generate_features(self):
        lightcurves = self._create_lightcurve_dataframe(
            copy.deepcopy(self.detections)
        )
        lightcurves = self.preprocessor.preprocess(lightcurves)
        metadata = self._create_metadata_dataframe(lightcurves)
        input_snids = lightcurves.index.unique().values

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

    def _create_metadata_dataframe(self, detections: pd.DataFrame):
        metadata = []
        for ef in detections.extra_fields:
            metadata.append(ef["diaObject"])
        metadata = pd.DataFrame.from_records(metadata)
        metadata.set_index(detections.index, inplace=True)
        metadata = metadata[~metadata.index.duplicated(keep="first")]
        return metadata
