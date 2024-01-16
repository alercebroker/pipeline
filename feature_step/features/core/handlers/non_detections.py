from typing import Callable, Literal

import methodtools
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from ._base import BaseHandler


class NonDetectionsHandler(BaseHandler):
    """Class for handling non-detections.

    Criteria for uniqueness is based on `id` (or `oid`, depending on use of `legacy`), `fid` and `mjd`.

    Required fields are `id`, `sid`, `fid`, `mjd` and `diffmaglim`.

    Additional keyword argument:
        `first_mjd` (pd.Series):  MJD of first detection. Should be indexed by `id` (and, optionally, `fid`)
    """

    _NAME = "non-detections"
    UNIQUE = ["id", "fid", "mjd"]
    NON_DUPLICATE = ("oid", "candid")
    COLUMNS = BaseHandler.COLUMNS + ["diffmaglim"]

    def _post_process(self, **kwargs):
        """Extracts MJD of first detection from `kwargs` in addition to base post-processing"""
        self.__first_mjd: pd.Series = kwargs.pop("first_mjd")
        self.__first_mjd = self.__first_mjd.fillna(
            self._alerts.groupby(["id", "fid"])["mjd"].max() + 1
        )
        super()._post_process(**kwargs)

    @methodtools.lru_cache()
    def alerts_when(
        self,
        *,
        when: Literal["before", "after"],
        surveys: str | tuple[str, ...] = (),
        bands: str | tuple[str, ...] = (),
    ) -> pd.DataFrame:
        """Get all alerts before or after the first detection

        Args:
            when: Either `before` or `after`
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all

        Returns:
            pd.DataFrame: Alerts before or after the date of first detection
        """
        if when == "before":
            func = lambda x: x.lt(self.__first_mjd.loc[x.name])
        elif when == "after":
            func = lambda x: x.gt(self.__first_mjd.loc[x.name])
        else:
            raise ValueError(f"Unrecognized value for 'when': {when}")

        by_fid = "fid" in self.__first_mjd.index.names
        mask = self.grouped(by_fid=by_fid, surveys=surveys, bands=bands)[
            "mjd"
        ].transform(func)
        if mask.any():
            return self._alerts[mask]
        columns = [
            c
            for c in self._alerts.columns
            if c not in self.__first_mjd.index.names
        ]
        return pd.DataFrame(
            np.nan, columns=columns, index=self.__first_mjd.index
        ).reset_index()

    @methodtools.lru_cache()
    def grouped_when(
        self,
        *,
        when: Literal["before", "after"],
        by_fid: bool = False,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
    ) -> DataFrameGroupBy:
        """Get alerts before or after first detection grouped by object (adn, optionally, band).

        Args:
            when: Either `before` or `after`
            by_fid: Whether to also group by band
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all

        Returns:
            DataFrameGroupBy: Grouped alerts
        """
        group = ["id", "fid"] if by_fid else "id"
        return self.alerts_when(
            when=when, surveys=surveys, bands=bands
        ).groupby(group)

    @methodtools.lru_cache()
    def which_index_when(
        self,
        *,
        which: Literal["first", "last"],
        when: Literal["before", "after"],
        by_fid: bool = False,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
    ) -> pd.Series:
        """Get first or last index of alerts before or after first detection for each object.

        Args:
            which: Either `first` or `last`
            when: Either `before` or `after`
            by_fid: Whether to get first or last index by band as well
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all

        Returns:
            pd.Series: First or last alert index. Indexed by `id` (and `fid` if `by_fid`).
        """
        if which not in ("first", "last"):
            raise ValueError(
                f"Unrecognized value for 'which': {which} (can only be first or last)"
            )
        function = "idxmin" if which == "first" else "idxmax"
        return self.agg_when(
            "mjd",
            function,
            when=when,
            by_fid=by_fid,
            surveys=surveys,
            bands=bands,
        )

    @methodtools.lru_cache()
    def which_value_when(
        self,
        column: str,
        *,
        which: Literal["first", "last"],
        when: Literal["before", "after"],
        by_fid: bool = False,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
    ) -> pd.Series:
        """Get first or last field value of alerts before or after first detection for each object.

        Args:
            column: Field for which to select values
            which: Either `first` or `last`
            when: Either `before` or `after`
            by_fid: Whether to get first or last value by band as well
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all

        Returns:
            pd.Series: First or last field value. Indexed by `id` (and `fid` if `by_fid`).
        """
        idx = self.which_index_when(
            which=which, when=when, by_fid=by_fid, surveys=surveys, bands=bands
        )
        df = self.alerts_when(when=when, surveys=surveys, bands=bands)
        if idx.isna().all():  # Happens for empty non-detections
            return idx
        return df[column][idx].set_axis(idx.index)

    @methodtools.lru_cache()
    def agg_when(
        self,
        column: str,
        func: str | Callable,
        *,
        when: Literal["before", "after"],
        by_fid: bool = False,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
    ) -> pd.Series:
        """Get aggregate alert field values before or after first detection.

        Args:
            column: Field over which the aggregation is performed
            func: Method name or callable that performs the aggregation per object
            when: Either `before` or `after`
            by_fid: Whether to aggregate by band as well
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all

        Returns:

        """
        return self.grouped_when(
            when=when, by_fid=by_fid, surveys=surveys, bands=bands
        )[column].agg(func)
