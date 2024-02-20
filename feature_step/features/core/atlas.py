from typing import List
import pandas as pd
from importlib import metadata as pymetadata


class AtlasFeatureExtractor:
    NAME = "atlas_lc_features"
    VERSION = pymetadata.version("feature-step")
    SURVEYS = ("ATLAS",)

    def __init__(
        self,
        detections: List[dict],
        non_detections: List[dict],
        xmatch: List[dict],
        **kwargs,
    ):
        pass

    def generate_features(self):
        return pd.DataFrame()
