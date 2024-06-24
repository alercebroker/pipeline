from typing import Union, Literal

import numpy as np
import pandas as pd

from ._base import BaseStatistics


class ObjectStatistics(BaseStatistics):
    _JOIN = "oid"

    @staticmethod
    def _arcsec2deg(
        values: Union[pd.Series, float]
    ) -> Union[pd.Series, float]:
        return values / 3600.0

    @staticmethod
    def _deg2arcsec(
        values: Union[pd.Series, float]
    ) -> Union[pd.Series, float]:
        return values * 3600.0

    @staticmethod
    def _compute_weights(
        sigmas: Union[pd.Series, float]
    ) -> Union[pd.Series, float]:
        return (
            sigmas.astype(float) ** -2
        )  # Integers cannot be raised to negative powers

    @classmethod
    def _weighted_mean(cls, values: pd.Series, sigmas: pd.Series) -> float:
        return np.average(values, weights=cls._compute_weights(sigmas))

    @classmethod
    def _weighted_mean_error(cls, sigmas: pd.Series) -> float:
        return np.sqrt(1 / np.sum(cls._compute_weights(sigmas)))

    def _calculate_coordinates(
        self, label: Literal["ra", "dec"]
    ) -> pd.DataFrame:
        def average(series):  # Needs wrapper to use the sigmas in the agg call
            return self._weighted_mean(series, sigmas.loc[series.index])

        sigmas = self._arcsec2deg(self._detections[f"e_{label}"])
        grouped_sigmas = self._group(sigmas.set_axis(self._detections["oid"]))
        return pd.DataFrame(
            {
                f"mean{label}": self._grouped_detections()[label].agg(average),
                f"sigma{label}": self._deg2arcsec(
                    grouped_sigmas.agg(self._weighted_mean_error)
                ),
            }
        )

    def _calculate_unique(self, label: str) -> pd.DataFrame:
        return pd.DataFrame(
            {label: self._grouped_detections()[label].unique().apply(list)}
        )

    def calculate_ra(self) -> pd.DataFrame:
        return self._calculate_coordinates("ra")

    def calculate_dec(self) -> pd.DataFrame:
        return self._calculate_coordinates("dec")

    def calculate_firstmjd(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"firstmjd": self._grouped_value("mjd", which="first")}
        )

    def calculate_lastmjd(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"lastmjd": self._grouped_value("mjd", which="last")}
        )

    def calculate_deltajd(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "deltajd": self._grouped_value("mjd", which="last")
                - self._grouped_value("mjd", which="first")
            }
        )

    def calculate_oid(self) -> pd.DataFrame:
        return self._calculate_unique("oid")

    def calculate_tid(self) -> pd.DataFrame:
        return self._calculate_unique("tid")

    def calculate_sid(self) -> pd.DataFrame:
        return self._calculate_unique("sid")

    def calculate_corrected(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "corrected": self._grouped_value(
                    "corrected", which="first", surveys=self._CORRECTED
                )
            }
        )

    def calculate_stellar(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "stellar": self._grouped_value(
                    "stellar", which="first", surveys=self._STELLAR
                )
            }
        )

    def calculate_ndet(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ndet": self._detections.value_counts(
                    subset=self._JOIN, sort=False
                )
            }
        )

    def calculate_reference_change(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "reference_change": self._grouped_value(
                    "mjdendref", which="last"
                )
                > self._grouped_value("mjd", which="first")
            }
        )

    def calculate_diffpos(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"diffpos": self._grouped_value("isdiffpos", which="first") > 0}
        )
