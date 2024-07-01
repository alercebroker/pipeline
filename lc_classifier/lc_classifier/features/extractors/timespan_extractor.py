import pandas as pd

from ..core.base import FeatureExtractor, AstroObject
import numpy as np


class TimespanExtractor(FeatureExtractor):
    def __init__(self):
        self.version = "1.0.0"

    def compute_features_single_object(self, astro_object: AstroObject):
        detections = astro_object.detections
        timespan = detections["mjd"].max() - detections["mjd"].min()
        sids = detections["sid"].unique()
        sids = np.sort(sids)
        sid = ",".join(sids)

        timespan_feature = pd.DataFrame(
            data=[["Timespan", timespan, None, sid, self.version]],
            columns=["name", "value", "fid", "sid", "version"],
        )
        all_features = [astro_object.features, timespan_feature]
        astro_object.features = pd.concat(
            [f for f in all_features if not f.empty], axis=0
        )
