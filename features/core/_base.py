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
    USE_CORRECTED: bool = False
    MIN_DETECTIONS: int = 0
    MIN_DETECTIONS_IN_FID: int = 0

    def __init__(self, detections: list[dict], non_detections: list[dict] | None = None):  # Should include xmatch info
        kwargs = dict(extras=self.EXTRAS, corrected=self.USE_CORRECTED)
        self.detections = DetectionsHandler(detections, surveys=self.SURVEYS, bands=self.BANDS, **kwargs)

        if self.MIN_DETECTIONS:
            self.detections.remove_objects(self.MIN_DETECTIONS)
        if self.MIN_DETECTIONS_IN_FID:
            self.detections.remove_objects(self.MIN_DETECTIONS_IN_FID, by_fid=True)

        self._discard_detections()

        first_mjd = self.detections.get_aggregate("mjd", "min", by_fid=True)
        non_detections = non_detections or []
        self.non_detections = NonDetectionsHandler(non_detections, surveys=self.SURVEYS, bands=self.BANDS, first_mjd=first_mjd)
        self.non_detections.match_objects(self.detections)

    @abc.abstractmethod
    def _discard_detections(self):
        pass

    def generate_features(self, exclude: set[str] | None = None) -> pd.DataFrame:
        exclude = exclude or set()  # Empty default
        # Add prefix to exclude, unless already provided
        exclude = {name if name.startswith(self._PREFIX) else f"{self._PREFIX}{name}" for name in exclude}

        # Select all methods that start with prefix unless excluded
        methods = {name for name in dir(self) if name.startswith(self._PREFIX) and name not in exclude}

        # Compute all features and join into single dataframe
        return pd.concat([getattr(self, method)() for method in methods], axis=1)
