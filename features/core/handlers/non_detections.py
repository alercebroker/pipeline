from typing import Literal

import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from ._base import BaseAlertHandler


class NonDetectionsHandler(BaseAlertHandler):
    UNIQUE = ["oid", "fid", "mjd"]
    NON_NULL_COLUMNS = BaseAlertHandler.NON_NULL_COLUMNS + ["diffmaglim"]

    def _get_alerts_when(
        self,
        mjd: pd.Series,
        *,
        when: Literal["before", "after"],
        surveys: str | tuple[str, ...] = (),
        bands: str | tuple[str, ...] = (),
    ):
        """`mjd` must be a series with indexes shared (by name) with non-detections (typically `aid`/`fid`)"""
        if when not in ("before", "after"):
            raise ValueError(f"Unrecognized value for 'when': {when}")
        non_detections = self._get_alerts(surveys=surveys, bands=bands).reset_index().set_index(mjd.index.names)
        mask = non_detections["mjd"] - mjd < 0 if when == "before" else non_detections["mjd"] - mjd > 0
        if mask.any():  # There's at least one non-detection in the range
            if self.INDEX:
                return non_detections[mask].reset_index().set_index(self.INDEX)
            return non_detections[mask].reset_index()
        return pd.DataFrame(np.nan, columns=non_detections.columns, index=mjd.index).reset_index()

    def get_grouped_when(
        self,
        mjd: pd.Series,
        *,
        when: Literal["before", "after"],
        by_fid: bool = False,
        surveys: str | tuple[str, ...] = (),
        bands: str | tuple[str, ...] = (),
    ) -> DataFrameGroupBy:
        group = ["aid", "fid"] if by_fid else "aid"
        return self._get_alerts_when(mjd, when=when, surveys=surveys, bands=bands).groupby(group)

    def get_which_index_when(
        self,
        mjd: pd.Series,
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
        return self.get_grouped_when(mjd, when=when, by_fid=by_fid, surveys=surveys, bands=bands)["mjd"].agg(function)

    def get_which_value_when(
        self,
        mjd: pd.Series,
        column: str,
        *,
        which: Literal["first", "last"],
        when: Literal["before", "after"],
        by_fid: bool = False,
        surveys: str | tuple[str, ...] = (),
        bands: str | tuple[str, ...] = (),
    ) -> pd.Series:
        idx = self.get_which_index_when(mjd, which=which, when=when, by_fid=by_fid, surveys=surveys, bands=bands)
        df = self._get_alerts_when(mjd, when=when, surveys=surveys, bands=bands)
        return df[column][idx].set_axis(idx.index)

    def get_aggregate_when(
        self, mjd: pd.Series, column: str, func: str, *, when: Literal["before", "after"], by_fid: bool = False
    ) -> pd.Series:
        return self.get_grouped_when(mjd, when=when, by_fid=by_fid)[column].agg(func)
