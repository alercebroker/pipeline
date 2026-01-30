import abc
from functools import reduce
from typing import Union, Literal, List, Set, Tuple

import pandas as pd
from methodtools import lru_cache
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy


class BaseStatistics(abc.ABC):
    """Base class for all statistics calculations"""
    _JOIN: Union[str, List[str]]
    _PREFIX = "calculate_"
    

    def __init__(self, detections: List[dict]):
        self._detections = pd.DataFrame.from_records(detections)
        # drop duplicate detections (same candid and oid)
        self._detections =  self._detections.sort_values(["new"], ascending=[False])
        self._detections = self._detections.drop_duplicates(["measurement_id", "oid"], keep="first")
        self._detections = self._detections.set_index(
            self._detections["measurement_id"].astype(str)
            + "_"
            + self._detections["oid"].astype(str)
        )
    
    @classmethod
    def _group(
        cls, df: Union[pd.DataFrame, pd.Series]
    ) -> Union[DataFrameGroupBy, SeriesGroupBy]:
        return df.groupby(cls._JOIN, dropna=False)
    
    @lru_cache(6)
    def _select_detections(
        self, *, surveys: Tuple[str] = None, corrected: bool = False
    ) -> pd.DataFrame:
        mask = (
            self._detections["corrected"]
            if corrected
            else pd.Series(True, index=self._detections.index)
        )
        return self._detections[mask]
    
    @lru_cache(12)
    def _grouped_index(
        self,
        *,
        which: Literal["first", "last"],
        surveys: Tuple[str] = None,
        corrected: bool = False,
    ) -> pd.Series:
        if which == "first":
            function = "idxmin"
        elif which == "last":
            function = "idxmax"
        else:
            raise ValueError(f"Unrecognized value for 'which': {which}")
        return self._grouped_detections(surveys=surveys, corrected=corrected)[
            "mjd"
        ].agg(function)
    
    @lru_cache(36)
    def _grouped_value(
        self,
        column: str,
        *,
        which: Literal["first", "last"],
        surveys: Tuple[str] = None,
        corrected: bool = False,
    ) -> pd.Series:
        idx = self._grouped_index(
            which=which, surveys=surveys, corrected=corrected
        )
        df = self._select_detections(surveys=surveys, corrected=corrected)
        return df[column][idx].set_axis(idx.index)
    
    @lru_cache(6)
    def _grouped_detections(
        self, *, surveys: Tuple[str] = None, corrected: bool = False
    ) -> DataFrameGroupBy:
        return self._group(
            self._select_detections(surveys=surveys, corrected=corrected)
        )
    
    def generate_statistics(self, exclude: Set[str] = None) -> pd.DataFrame:
        exclude = exclude or set()  # Empty default
        # Add prefix to exclude, unless already provided
        exclude = {
            name if name.startswith(self._PREFIX) else f"{self._PREFIX}{name}"
            for name in exclude
        }
        # Select all methods that start with prefix unless excluded
        methods = {
            name
            for name in dir(self)
            if name.startswith(self._PREFIX) and name not in exclude
        }
        # Compute all statistics and join into single dataframe
        stats = [getattr(self, method)() for method in methods]
        return reduce(lambda left, right: left.join(right, how="outer"), stats)
