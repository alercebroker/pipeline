import abc
from functools import reduce
from typing import Any

import pandas as pd

from .handlers import DetectionsHandler, NonDetectionsHandler


class BaseFeatureExtractor(abc.ABC):
    _PREFIX: str = "calculate_"
    SURVEYS: str | tuple[str, ...] = ()
    BANDS: str | tuple[str, ...] = ()
    BANDS_MAPPING: dict[str, Any] = {}
    EXTRAS: list[str] = []
    MIN_DETECTIONS: int = 0
    MIN_DETECTIONS_IN_FID: int = 0

    def __init__(self, detections: list[dict], non_detections: list[dict] | None = None):  # Should include xmatch info
        self.detections = DetectionsHandler(detections, surveys=self.SURVEYS, bands=self.BANDS, extras=self.EXTRAS)
        self.non_detections = NonDetectionsHandler(non_detections or [], surveys=self.SURVEYS, bands=self.BANDS)

        self.detections.remove_objects(self.MIN_DETECTIONS)
        self.detections.remove_objects(self.MIN_DETECTIONS_IN_FID, by_fid=True)

        self._discard_detections()

        self.non_detections.match_objects(self.detections)

    def _discard_detections(self):
        pass

    def generate_features(self, exclude: set[str] | None = None) -> pd.DataFrame:
        exclude = exclude or set()  # Empty default
        # Add prefix to exclude, unless already provided
        exclude = {name if name.startswith(self._PREFIX) else f"{self._PREFIX}{name}" for name in exclude}

        # Select all methods that start with prefix unless excluded
        methods = {name for name in dir(self) if name.startswith(self._PREFIX) and name not in exclude}

        # Compute all features and join into single dataframe
        feats = [getattr(self, method)() for method in methods]
        return reduce(lambda left, right: left.join(right, how="outer"), feats)
