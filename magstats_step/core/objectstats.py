from typing import Union, Literal

import numpy as np
import pandas as pd

from .magstats import MagnitudeStatistics


class ObjectStatistics:
    CALCULATOR_PREFIX = "calculate_"
    EXTRA_COLUMNS = ["distpsnr1", "sgscore1", "chinr", "sharpnr"]

    def __init__(self, aid: str, detections: dict, non_detections: dict, exclude: Union[set, None] = None):
        self._aid = aid
        self._detections = pd.DataFrame.from_records(detections, exclude=["extra_fields"], index="candid")
        if non_detections:
            self._non_detections = pd.DataFrame.from_records(non_detections)
        else:
            self._non_detections = pd.DataFrame()

        exclude = exclude or set()
        self._exclude = {f"{self.CALCULATOR_PREFIX}{n}" for n in exclude if not n.startswith(self.CALCULATOR_PREFIX)}

    @staticmethod
    def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        return np.average(values, weights=weights)

    @staticmethod
    def weighted_error(weights: pd.Series) -> float:
        return np.sqrt(1 / np.sum(weights))

    @staticmethod
    def arcsec2dec(values: Union[pd.Series, float]) -> Union[pd.Series, float]:
        return values / 3600.

    @staticmethod
    def dec2arcsec(values: Union[pd.Series, float]) -> Union[pd.Series, float]:
        return values * 3600.

    def _calculate_coordinates(self, label: Literal["ra", "dec"]) -> dict:
        weights = 1 / self.arcsec2dec(self._detections[f"e_{label}"]) ** 2
        return {
            f"mean{label}": self.weighted_mean(self._detections[label], weights),
            f"sigma{label}": self.dec2arcsec(self.weighted_error(weights))
        }

    def _calculate_unique(self, label: str) -> dict:
        return {label: list(self._detections[label].unique())}

    def calculate_coordinates(self) -> dict:
        ra = self._calculate_coordinates("ra")
        dec = self._calculate_coordinates("dec")
        return {**ra, **dec}

    def calculate_ndet(self) -> dict:
        return {"ndet": len(self._detections.index)}

    def calculate_mjd(self) -> dict:
        return {
            "firstmjd": self._detections.mjd.min(),
            "lastmjd": self._detections.mjd.max()
        }

    def calculate_oid(self) -> dict:
        return self._calculate_unique("oid")

    def calculate_tid(self) -> dict:
        return self._calculate_unique("tid")

    def calculate_corrected(self) -> dict:
        idx = self._detections["mjd"].idxmin()
        return {"corrected": self._detections["corrected"][idx]}

    def calculate_magstats(self) -> dict:
        calculator = MagnitudeStatistics(self._detections, self._non_detections, self._exclude)
        return {"magstats": calculator.generate_magstats()}

    def generate_object(self) -> dict:
        methods = [m for m in ObjectStatistics.__dict__ if m.startswith("calculate_") and m not in self._exclude]
        return {k: v for method in methods for k, v in getattr(self, method)().items()}
