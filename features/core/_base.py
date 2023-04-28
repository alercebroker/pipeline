import abc
from typing import Any

import pandas as pd

from .handlers import DetectionsHandler, NonDetectionsHandler


class BaseFeatureExtractor(abc.ABC):
    _PREFIX: str = "calculate_"
    SURVEYS: str | tuple[str, ...] = ()
    BANDS: str | tuple[str, ...] = ()
    BANDS_MAPPING: dict[str, Any] = {}
    EXTRA_COLUMNS: list[str] = []
    XMATCH_COLUMNS: list[str] = []
    USE_CORRECTED: bool = False
    MIN_DETECTIONS: int = 0
    MIN_DETECTIONS_IN_FID: int = 0

    def __init__(
        self,
        detections: list[dict] | pd.DataFrame,
        non_detections: list[dict] | pd.DataFrame,
        xmatches: list[dict] | pd.DataFrame = None,
        *,
        legacy: bool = False
    ):
        common = dict(surveys=self.SURVEYS, bands=self.BANDS, legacy=legacy)

        if isinstance(detections, pd.DataFrame):
            detections = detections.reset_index().to_dict("records")
        self.detections = DetectionsHandler(detections, extras=self.EXTRA_COLUMNS, corr=self.USE_CORRECTED, **common)
        self._discard_detections()

        if isinstance(non_detections, pd.DataFrame):
            non_detections = non_detections.reset_index().to_dict("records")
        first_mjd = self.detections.get_aggregate("mjd", "min", by_fid=True)
        self.non_detections = NonDetectionsHandler(non_detections, first_mjd=first_mjd, **common)
        self.non_detections.match(self.detections)

        if isinstance(xmatches, pd.DataFrame):
            xmatches = xmatches.reset_index().to_dict("records")
        self.xmatches = self._create_xmatches(xmatches, legacy)
        self._periods = None

    @abc.abstractmethod
    def _discard_detections(self):
        self.detections.remove_objects_without_enough_detections(self.MIN_DETECTIONS)
        self.detections.remove_objects_without_enough_detections(self.MIN_DETECTIONS_IN_FID, by_fid=True)

    def _create_xmatches(self, xmatches: list[dict], legacy: bool) -> pd.DataFrame:
        id_col = "oid" if legacy else "aid"
        xmatches = pd.DataFrame(xmatches) if xmatches else pd.DataFrame(columns=[id_col] + self.XMATCH_COLUMNS)
        xmatches = xmatches.rename(columns={id_col: "id"})
        xmatches = xmatches[xmatches["id"].isin(self.detections.get_objects())]
        return xmatches.set_index("id")[self.XMATCH_COLUMNS]

    def clear_caches(self):
        self.detections.clear_caches()
        self.non_detections.clear_caches()

    def generate_features(self, exclude: set[str] | None = None) -> pd.DataFrame:
        exclude = exclude or set()  # Empty default
        # Add prefix to exclude, unless already provided
        exclude = {name if name.startswith(self._PREFIX) else f"{self._PREFIX}{name}" for name in exclude}

        # Select all methods that start with prefix unless excluded
        methods = {name for name in dir(self) if name.startswith(self._PREFIX) and name not in exclude}

        # Compute all features and join into single dataframe
        return pd.concat((getattr(self, method)() for method in methods), axis="columns", copy=False)
