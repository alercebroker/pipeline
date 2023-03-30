from typing import Union, Literal, List

import numpy as np
import pandas as pd

from ._base import BaseStatistics


class ObjectStatistics(BaseStatistics):
    _JOIN = "aid"

    def __init__(self, detections: List[dict]):
        super().__init__(detections)

    @staticmethod
    def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        return np.average(values, weights=weights)

    @staticmethod
    def weighted_error(weights: pd.Series) -> float:
        return np.sqrt(1 / np.sum(weights))

    @staticmethod
    def arcsec2dec(values: Union[pd.Series, float]) -> Union[pd.Series, float]:
        return values / 3600.0

    @staticmethod
    def dec2arcsec(values: Union[pd.Series, float]) -> Union[pd.Series, float]:
        return values * 3600.0

    def _calculate_coordinates(self, label: Literal["ra", "dec"]) -> dict:
        def _average(series):
            return self.weighted_mean(series, weights.loc[series.index])

        weights = 1 / self.arcsec2dec(self._detections[f"e_{label}"]) ** 2
        weights_by_aid = weights.set_axis(self._detections["aid"])
        return {
            f"mean{label}": self._grouped_detections()[label].agg(_average),
            f"sigma{label}": self.dec2arcsec(self._group(weights_by_aid).agg(self.weighted_error)),
        }

    def _calculate_unique(self, label: str) -> dict:
        return {label: self._grouped_detections()[label].unique()}

    def calculate_ra(self) -> pd.DataFrame:
        return pd.DataFrame(self._calculate_coordinates("ra"))

    def calculate_dec(self) -> pd.DataFrame:
        return pd.DataFrame(self._calculate_coordinates("dec"))

    def calculate_ndet(self) -> pd.DataFrame:
        # The column selected for ndet is irrelevant as long as it has no NaN values
        return pd.DataFrame({"ndet": self._grouped_detections()["oid"].count()})

    def calculate_firstmjd(self) -> pd.DataFrame:
        return pd.DataFrame({"firstmjd": self._grouped_value("mjd", which="first")})

    def calculate_lastmjd(self) -> pd.DataFrame:
        return pd.DataFrame({"lastmjd": self._grouped_value("mjd", which="last")})

    def calculate_oid(self) -> pd.DataFrame:
        return pd.DataFrame(self._calculate_unique("oid"))

    def calculate_tid(self) -> pd.DataFrame:
        return pd.DataFrame(self._calculate_unique("tid"))

    def calculate_corrected(self) -> pd.DataFrame:
        return pd.DataFrame({"corrected": self._grouped_value("corrected", which="first", surveys=self._CORRECTED)})

    def calculate_stellar(self) -> pd.DataFrame:
        return pd.DataFrame({"stellar": self._grouped_value("stellar", which="first", surveys=self._STELLAR)})
