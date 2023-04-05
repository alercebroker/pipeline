from typing import Union, Literal, List

import numpy as np
import pandas as pd

from ._base import BaseStatistics


class ObjectStatistics(BaseStatistics):
    _JOIN = "aid"

    def __init__(self, detections: List[dict]):
        super().__init__(detections)

    @staticmethod
    def _compute_weights(sigmas: pd.Series) -> pd.Series:
        return sigmas.astype(float) ** -2  # Integers cannot be raised to negative powers

    @classmethod
    def _weighted_mean(cls, values: pd.Series, sigmas: pd.Series) -> float:
        return np.average(values, weights=cls._compute_weights(sigmas))

    @classmethod
    def _weighted_mean_error(cls, sigmas: pd.Series) -> float:
        return np.sqrt(1 / np.sum(cls._compute_weights(sigmas)))

    @staticmethod
    def _arcsec2dec(values: Union[pd.Series, float]) -> Union[pd.Series, float]:
        return values / 3600.0

    @staticmethod
    def _dec2arcsec(values: Union[pd.Series, float]) -> Union[pd.Series, float]:
        return values * 3600.0

    def _calculate_coordinates(self, label: Literal["ra", "dec"]) -> pd.DataFrame:
        def _average(series):
            return self._weighted_mean(series, sigmas.loc[series.index])

        sigmas = self._arcsec2dec(self._detections[f"e_{label}"])
        grouped_sigmas = self._group(sigmas.set_axis(self._detections["aid"]))
        return pd.DataFrame(
            {
                f"mean{label}": self._grouped_detections()[label].agg(_average),
                f"sigma{label}": self._dec2arcsec(grouped_sigmas.agg(self._weighted_mean_error)),
            }
        )

    def _calculate_unique(self, label: str) -> pd.DataFrame:
        return pd.DataFrame({label: self._grouped_detections()[label].unique()})

    def calculate_ra(self) -> pd.DataFrame:
        return self._calculate_coordinates("ra")

    def calculate_dec(self) -> pd.DataFrame:
        return self._calculate_coordinates("dec")

    def calculate_ndet(self) -> pd.DataFrame:
        return super().calculate_ndet()

    def calculate_firstmjd(self) -> pd.DataFrame:
        return pd.DataFrame({"firstmjd": self._grouped_value("mjd", which="first")})

    def calculate_lastmjd(self) -> pd.DataFrame:
        return pd.DataFrame({"lastmjd": self._grouped_value("mjd", which="last")})

    def calculate_oid(self) -> pd.DataFrame:
        return self._calculate_unique("oid")

    def calculate_tid(self) -> pd.DataFrame:
        return self._calculate_unique("tid")

    def calculate_corrected(self) -> pd.DataFrame:
        return pd.DataFrame({"corrected": self._grouped_value("corrected", which="first", surveys=self._CORRECTED)})

    def calculate_stellar(self) -> pd.DataFrame:
        return pd.DataFrame({"stellar": self._grouped_value("stellar", which="first", surveys=self._STELLAR)})
