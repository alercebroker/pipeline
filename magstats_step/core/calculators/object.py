from typing import Union, Literal

import numpy as np
import pandas as pd

from .magstats import MagnitudeStatistics


class ObjectStatistics:
    EXTRA_COLUMNS = ["distpsnr1", "sgscore1", "chinr", "sharpnr"]

    def __init__(self, aid: str, detections: dict, non_detections: dict):
        self._aid = aid
        extras = [{"candid": det["candid"], **det["extra_fields"]} for det in detections]
        detections = pd.DataFrame.from_records(detections, exclude=["extra_fields"], index="candid")
        extras = pd.DataFrame.from_records(extras, columns=self.EXTRA_COLUMNS + ["candid"], index="candid")

        self._non_detections = pd.DataFrame.from_records(non_detections)
        self._detections = detections.join(extras)

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
        return {"magstats": MagnitudeStatistics(self._detections).calculate()}

    def calculate_all(self, exclude: Union[set, None] = None) -> dict:
        exclude = exclude or set()
        exclude = {f"calculate_{name}" for name in exclude if not name.startswith("calculate_")}

        compute = [method for method in self.__dict__ if method.startswith("calculate_") and method not in exclude]
        return {k: v for method in compute for k, v in getattr(self, method)().items()}
