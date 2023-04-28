from typing import Literal

import methodtools
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from ._base import BaseHandler


class NonDetectionsHandler(BaseHandler):
    UNIQUE = ["oid", "fid", "mjd"]
    _COLUMNS = BaseHandler._COLUMNS + ["diffmaglim"]

    def _post_process_alerts(self, **kwargs):
        self.__first_mjd: pd.Series = kwargs.pop("first_mjd")
        super()._post_process_alerts(**kwargs)

    @methodtools.lru_cache()
    def _get_alerts_when(
        self,
        *,
        when: Literal["before", "after"],
        surveys: str | tuple[str, ...] = (),
        bands: str | tuple[str, ...] = (),
    ):
        """`mjd` must be a series with indexes shared (by name) with non-detections (typically `aid`/`fid`)"""
        if when == "before":
            func = lambda x: x.lt(self.__first_mjd.loc[x.name])
        elif when == "after":
            func = lambda x: x.gt(self.__first_mjd.loc[x.name])
        else:
            raise ValueError(f"Unrecognized value for 'when': {when}")

        by_fid = "fid" in self.__first_mjd.index.names
        mask = self.get_grouped(by_fid=by_fid, surveys=surveys, bands=bands)["mjd"].transform(func)
        if mask.any():
            return self._alerts[mask]
        columns = [c for c in self._alerts.columns if c not in self.__first_mjd.index.names]
        return pd.DataFrame(np.nan, columns=columns, index=self.__first_mjd.index).reset_index()

    @methodtools.lru_cache()
    def get_grouped_when(
        self,
        *,
        when: Literal["before", "after"],
        by_fid: bool = False,
        surveys: str | tuple[str, ...] = (),
        bands: str | tuple[str, ...] = (),
    ) -> DataFrameGroupBy:
        group = ["aid", "fid"] if by_fid else "aid"
        return self._get_alerts_when(when=when, surveys=surveys, bands=bands).groupby(group)

    @methodtools.lru_cache()
    def get_which_index_when(
        self,
        *,
        which: Literal["first", "last"],
        when: Literal["before", "after"],
        by_fid: bool = False,
        surveys: str | tuple[str, ...] = (),
        bands: str | tuple[str, ...] = (),
    ) -> pd.Series:
        match which:
            case "first":
                function = "idxmin"
            case "last":
                function = "idxmax"
            case _:
                raise ValueError(f"Unrecognized value for 'which': {which} (can only be first or last)")
        return self.get_grouped_when(when=when, by_fid=by_fid, surveys=surveys, bands=bands)["mjd"].agg(function)

    @methodtools.lru_cache()
    def get_which_value_when(
        self,
        column: str,
        *,
        which: Literal["first", "last"],
        when: Literal["before", "after"],
        by_fid: bool = False,
        surveys: str | tuple[str, ...] = (),
        bands: str | tuple[str, ...] = (),
    ) -> pd.Series:
        idx = self.get_which_index_when(which=which, when=when, by_fid=by_fid, surveys=surveys, bands=bands)
        df = self._get_alerts_when(when=when, surveys=surveys, bands=bands)
        if idx.isna().all():  # Happens for empty non-detections
            return idx
        return df[column][idx].set_axis(idx.index)

    @methodtools.lru_cache()
    def get_aggregate_when(
        self, column: str, func: str, *, when: Literal["before", "after"], by_fid: bool = False
    ) -> pd.Series:
        return self.get_grouped_when(when=when, by_fid=by_fid)[column].agg(func)
