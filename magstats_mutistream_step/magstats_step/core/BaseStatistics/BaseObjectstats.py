import pandas as pd
import numpy as np
from typing import Union, Literal

from ._base import BaseStatistics

class BaseObjectStatistics(BaseStatistics):
    """Base class for object statistics common to all surveys"""
    _JOIN = "oid"
    
    @staticmethod
    def _arcsec2deg(values: Union[pd.Series, float]) -> Union[pd.Series, float]:
        return values / 3600.0
    
    @staticmethod
    def _deg2arcsec(values: Union[pd.Series, float]) -> Union[pd.Series, float]:
        return values * 3600.0
    
    @staticmethod
    def _compute_weights(sigmas: Union[pd.Series, float]) -> Union[pd.Series, float]:
        return sigmas.astype(float) ** -2
    
    @classmethod
    def _weighted_mean(cls, values: pd.Series, sigmas: pd.Series) -> float:
        return np.average(values, weights=cls._compute_weights(sigmas))
    
    @classmethod
    def _weighted_mean_error(cls, sigmas: pd.Series) -> float:
        return np.sqrt(1 / np.sum(cls._compute_weights(sigmas)))
    
    def _calculate_coordinates(self, label: Literal["ra", "dec"]) -> pd.DataFrame:
        def average(series):
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
        grouped = self._grouped_detections()
        return grouped[label].agg(lambda x: x.iloc[0]).to_frame(name=label) 
    
    
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
        return pd.DataFrame({
            "deltajd": self._grouped_value("mjd", which="last")
                      - self._grouped_value("mjd", which="first")
        })
    
    def calculate_oid(self) -> pd.DataFrame:
        return self._calculate_unique("oid")
    
    def calculate_tid(self) -> pd.DataFrame:
        return self._calculate_unique("tid")
    
    def calculate_sid(self) -> pd.DataFrame:
        return self._calculate_unique("sid")
    
    def calculate_ndet(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ndet": self._detections.value_counts(subset=self._JOIN, sort=False)
        })
