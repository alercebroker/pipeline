import abc
from functools import reduce
from typing import Union, Literal, List, Set

import pandas as pd
from methodtools import lru_cache
from pandas.core.groupby import DataFrameGroupBy, SeriesGroupBy

Which = Literal["first", "last"]


class BaseStatistics(abc.ABC):
    _PREFIX = "calculate_"

    def __init__(self, detections: List[dict]):
        self._detections = pd.DataFrame.from_records(detections, exclude=["extra_fields"], index="candid")

    @staticmethod
    @abc.abstractmethod
    def _group(df: Union[pd.DataFrame, pd.Series]) -> Union[DataFrameGroupBy, SeriesGroupBy]:
        return df.groupby("aid")

    @lru_cache(1)
    def _corrected(self):
        return self._detections[self._detections["corrected"]]

    @lru_cache(4)
    def _grouped_index(self, which: Which, corrected: bool = False) -> pd.Series:
        if which == "first":
            function = "idxmin"
        elif which == "last":
            function = "idxmax"
        else:
            raise ValueError(f"Unrecognized value for 'which': {which}")
        return self._grouped_detections(corrected)["mjd"].agg(function)

    @lru_cache(10)
    def _grouped_value(self, source: str, which: Which, corrected: bool = False) -> pd.Series:
        idx = self._grouped_index(which, corrected)
        df = self._corrected() if corrected else self._detections
        return df[source][idx].set_axis(idx.index)

    @lru_cache(2)
    def _grouped_detections(self, corrected: bool = False) -> DataFrameGroupBy:
        return self._group(self._corrected()) if corrected else self._group(self._detections)

    def generate_statistics(self, exclude: Set[str] = None) -> pd.DataFrame:
        exclude = exclude or set()  # Empty default
        # Add prefix to exclude, unless already provided
        exclude = {name if name.startswith(self._PREFIX) else f"{self._PREFIX}{name}" for name in exclude}

        # Select all methods that start with prefix unless excluded
        methods = {name for name in self.__class__.__dict__ if name.startswith(self._PREFIX) and name not in exclude}

        # Compute all statistics and join into single dataframe
        stats = {getattr(self, method)() for method in methods}
        return reduce(lambda left, right: left.join(right, how="outer"), stats)
