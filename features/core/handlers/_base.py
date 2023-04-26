from __future__ import annotations

import abc
import functools
from typing import Literal

import pandas as pd
from pandas.core.groupby import DataFrameGroupBy


class BaseAlertHandler(abc.ABC):
    """Alerts require at the very least fields `sid`, `aid` and `mjd`"""

    INDEX: str | list[str] = []
    UNIQUE: str | list[str] = []
    NON_NULL_COLUMNS = ["aid", "sid", "fid", "mjd"]

    def __init__(self, alerts: list[dict], *, surveys: str | tuple[str] = (), bands: str | tuple[str] = (), **kwargs):
        self._alerts = self._create_df(alerts, **kwargs)
        self._clear(surveys=surveys, bands=bands)

    def remove_objects(self, minimum: int, *, by_fid: bool = False):
        """If using `by_fid` at least one band must exceed the minimum"""
        mask = self.get_grouped(by_fid=by_fid)["mjd"].count().groupby("aid").max() > minimum
        self._alerts = self._alerts[self._alerts["aid"].isin(mask.index[mask])]

    def match_objects(self, other: BaseAlertHandler):
        self._alerts = self._alerts[self._alerts["aid"].isin(other._alerts["aid"].unique())]

    def remove_alerts_out_of_range(
        self, column: str, *, lt: float = None, gt: float = None, le: bool = False, ge: bool = False
    ):
        mask = pd.Series(True, index=self._alerts.index, dtype=bool)
        series = self._alerts[column]
        if lt is not None:
            mask &= (series <= lt) if le else (series < lt)
        if gt is not None:
            mask &= (series >= gt) if ge else (series > gt)
        self._alerts = self._alerts[mask]

    @classmethod
    def _create_df(cls, alerts: list[dict], **kwargs) -> pd.DataFrame:
        if kwargs:
            # Only used in subclasses, when using super it should be empty
            raise ValueError(f"Unrecognized kwargs: {', '.join(kwargs)}")
        try:
            df = pd.DataFrame.from_records(alerts, exclude=["extra_fields"])
        except KeyError:  # extra_fields is not present
            df = pd.DataFrame.from_records(alerts)
        if df.size == 0:
            index = {cls.INDEX} if isinstance(cls.INDEX, str) else set(cls.INDEX)
            unique = {cls.UNIQUE} if isinstance(cls.UNIQUE, str) else set(cls.UNIQUE)

            columns = set(cls.NON_NULL_COLUMNS) | index | unique
            df = pd.DataFrame(columns=list(columns))

        if cls.UNIQUE:
            df.drop_duplicates(cls.UNIQUE, inplace=True)
        if cls.INDEX:
            df.set_index(cls.INDEX, inplace=True)

        return df

    def _clear(self, surveys: str | tuple[str, ...], bands: str | tuple[str, ...]):
        self.__discard_invalid_alerts()
        self.__discard_not_in_bands(bands)
        self.__discard_not_in_surveys(surveys)

    def __discard_invalid_alerts(self):
        with pd.option_context("mode.use_inf_as_na", True):
            self._alerts = self._alerts[self._alerts[self.NON_NULL_COLUMNS].notna().all(axis="columns")]

    def __discard_not_in_surveys(self, surveys: str | tuple[str]):
        self._alerts = self._alerts[self.__surveys_mask(surveys)]

    def __discard_not_in_bands(self, bands: str | tuple[str]):
        self._alerts = self._alerts[self.__bands_mask(bands)]

    def __mask(self, column: str, values: tuple[str]):
        if not values:
            return pd.Series(True, index=self._alerts.index)
        masks = (self._alerts[column].str.lower() == value.lower() for value in values)
        return functools.reduce(lambda l, r: l.__or__(r), masks)

    def __surveys_mask(self, surveys: str | tuple[str, ...] = ()) -> pd.Series:
        surveys = (surveys,) if isinstance(surveys, str) else surveys
        return self.__mask("sid", surveys)

    def __bands_mask(self, bands: str | tuple[str, ...] = ()) -> pd.Series:
        bands = (bands,) if isinstance(bands, str) else bands
        return self.__mask("fid", bands)

    def _get_alerts(self, *, surveys: str | tuple[str, ...] = (), bands: str | tuple[str, ...] = ()) -> pd.DataFrame:
        return self._alerts[self.__surveys_mask(surveys) & self.__bands_mask(bands)]

    def get_grouped(
        self, *, by_fid: bool = False, surveys: str | tuple[str, ...] = (), bands: str | tuple[str, ...] = ()
    ) -> DataFrameGroupBy:
        return self._get_alerts(surveys=surveys, bands=bands).groupby(["aid", "fid"] if by_fid else "aid")

    def get_which_index(
        self,
        *,
        which: Literal["first", "last"],
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
        return self.get_grouped(by_fid=by_fid, surveys=surveys, bands=bands)["mjd"].agg(function)

    def get_which_value(
        self,
        column: str,
        *,
        which: Literal["first", "last"],
        by_fid: bool = False,
        surveys: str | tuple[str, ...] = (),
        bands: str | tuple[str, ...] = (),
    ) -> pd.Series:
        idx = self.get_which_index(which=which, by_fid=by_fid, surveys=surveys, bands=bands)
        df = self._get_alerts(surveys=surveys, bands=bands)
        return df[column][idx].set_axis(idx.index)

    def get_aggregate(self, column: str, func: str, *, by_fid: bool = False, surveys: str | tuple[str, ...] = (),
                      bands: str | tuple[str, ...] = ()) -> pd.Series:
        return self.get_grouped(by_fid=by_fid, surveys=surveys, bands=bands)[column].agg(func)

    def get_delta(self, column: str, *, by_fid: bool = False, surveys: str | tuple[str, ...] = (), bands: str | tuple[str, ...] = ()) -> pd.Series:
        vmax = self.get_aggregate(column, "max", by_fid=by_fid, surveys=surveys, bands=bands)
        vmin = self.get_aggregate(column, "min", by_fid=by_fid, surveys=surveys, bands=bands)
        return vmax - vmin
