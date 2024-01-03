from lc_classifier.features.custom.elasticc_feature_extractor import (
    ElasticcFeatureExtractor,
)
from lc_classifier.features.preprocess.preprocess_elasticc import (
    ElasticcPreprocessor,
)
from typing import List
import pandas as pd
import pickle
import copy
from importlib import metadata as pymetadata


class ELAsTiCCFeatureExtractor:
    NAME = "elasticc_lc_features"
    VERSION = pymetadata.version("feature-step")
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
        input_snids = lightcurves.index.unique().values
        metadata = self._create_metadata_dataframe(lightcurves)
        metadata = self.preprocessor.preprocess_metadata(metadata)
        lightcurves = self.preprocessor.preprocess(lightcurves)

        features = self.extractor.compute_features(
            lightcurves, metadata=metadata, force_snids=input_snids
        )
        return features

    def _create_lightcurve_dataframe(self, detections: List[dict]):
        self._rename_detections_columns(detections)
        return pd.DataFrame.from_records(detections).set_index("oid")

    def _rename_detections_columns(self, detections: List[dict]):
        for det in detections:
            det["MJD"] = det.pop("mjd")
            det["FLUXCAL"] = det.pop("mag_corr")
            det["FLUXCALERR"] = det.pop("e_mag_corr")
            det["BAND"] = det.pop("fid")

    def _create_metadata_dataframe(self, detections: pd.DataFrame):
        # Keep one metadata row per object, as it should not change
        # between detections for a given object
        one_detection_per_object = detections[["extra_fields"]][
            ~detections.index.duplicated(keep="first")
        ]
        metadata = []
        for ef in one_detection_per_object["extra_fields"]:
            metadata.append(pickle.loads(ef["diaObject"])[0])
        metadata = pd.DataFrame.from_records(metadata)
        metadata.set_index(one_detection_per_object.index, inplace=True)
        return metadata
