from __future__ import annotations

import abc
import functools
from typing import Callable, Literal

import methodtools
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy


class BaseHandler(abc.ABC):
    """Base class for handling alerts, typically detections (including forced photometry) and non-detections.

    This provides the general base methods for extracting statistics and applying functions to the alerts
    considered.

    Alerts require at the very least fields `sid`, `fid`, `mjd` and `aid` (the last can be `oid` if using `legacy`).
    Usually, a subclass will define additional fields that are required, but missing one of the aforementioned fields
    will result in an initialization error, unless the initialization methods are adapted as well. These always
    required fields are defined the class attribute `COLUMNS`. The field `aid` (or `oid`) will be renamed to `id`.

    It is possible to define one or more fields that define a unique alert with the class attribute `UNIQUE`.
    Alert duplicates (based on these columns) will be eliminated, keeping only one of these alerts. It is also
    possible to define one or more fields for indexing the alerts using the class attribute `INDEX`. Both unique
    and index fields are optional and should be defined in subclasses.

    Note that there is heavy use of caching, so externally modifying the alerts within the class can cause wrong
    results to come up. For development, it is suggested that methods that can modify the internal data frame
    make use of the method `clear_caches` to ensure that subsequent calls on the main methods give consistent results.
    """

    INDEX: str | list[str] = []
    UNIQUE: str | list[str] = []
    COLUMNS = ["id", "sid", "fid", "mjd"]

    def __init__(self, alerts: list[dict], *, surveys: str | tuple[str] = (), bands: str | tuple[str] = (), **kwargs):
        """Initialize alerts.

        Here we will use "alert" to refer to either detections or non-detections, depending on specific implementation.

        This handler can receive a list of alerts from multiple objects simultaneously and will produce results for
        each object (and band, depending on the options provided when calling the specific method). Fields defined in
        `COLUMNS` must be present and have to contain defined values (i.e., no NaN, NA or `None`).

        To keep fields besides those defined in `COLUMNS`, at initialization the option `extras` can be passed as a
        list of strings. These are additional fields to be kept, if present directly in the alert, or taken from inside
        the `extra_fields` field of the alert if present. If not found, these extra columns will be filled with NaNs.

        Args:
            alerts: list of alerts, with each alert as a dictionary containing fields expected for ALeRCE alerts

        Keyword Args:
            surveys (str | tuple[str]): Survey(s) to keep. Defined by field `sid`. An empty tuple will keep all
            bands (str | tuple[str]): Band(s) to keep. Defined by `fid`. An empty tuple will keep all
            legacy (bool): Whether to use legacy format for alerts following old PSQL style database from ALeRCE
            extras (list[str]): Additional fields to keep (these fields can have undefined values)
        """

        legacy = kwargs.pop("legacy", False)
        try:
            self._alerts = pd.DataFrame.from_records(alerts, exclude=["extra_fields"])
        except KeyError:  # extra_fields is not present
            self._alerts = pd.DataFrame.from_records(alerts)
        self._alerts = self._alerts.rename(columns={"oid" if legacy else "aid": "id"})
        if self._alerts.size == 0:
            index = {self.INDEX} if isinstance(self.INDEX, str) else set(self.INDEX)
            unique = {self.UNIQUE} if isinstance(self.UNIQUE, str) else set(self.UNIQUE)

            columns = set(self.COLUMNS) | index | unique
            self._alerts = pd.DataFrame(columns=list(columns))

        if legacy:
            self._alerts["sid"] = "ZTF"
            self._alerts["fid"].replace({1: "g", 2: "r"}, inplace=True)

        if self.UNIQUE:
            self._alerts.drop_duplicates(self.UNIQUE, inplace=True)
        if self.INDEX:
            self._alerts.set_index(self.INDEX, inplace=True)

        extras = kwargs.get("extras", [])
        self._post_process_alerts(alerts=alerts, surveys=surveys, legacy=legacy, **kwargs)
        self._clear(surveys=surveys, bands=bands, extras=extras)

    def remove_objects_without_enough_detections(self, minimum: int, *, by_fid: bool = False):
        """Remove all alerts from objects without a minimum number of detections.

        Args:
            minimum: Minimum number of detections
            by_fid: If `True`, checks that at least one band has the specified minimum

        """
        self.clear_caches()
        if by_fid:
            mask = self._alerts.groupby("id")["fid"].value_counts().unstack("fid").max(axis="columns") > minimum
        else:
            mask = self._alerts.groupby("id")["fid"].count() > minimum
        self._alerts = self._alerts[self._alerts["id"].isin(mask.index[mask])]

    def match(self, other: BaseHandler):
        """Removes alerts from objects contained in the current handler that are not present in the other.

        Only modifies the current handler, the other one will not be modified.

        Args:
            other: Handler used to check for objects
        """
        self.clear_caches()
        self._alerts = self._alerts[self._alerts["id"].isin(other.get_objects())]

    def select(
        self, column: str | list[str], *, lt: float | list[float, None] = None, gt: float | list[float, None] = None
    ):
        """Removes all alerts outside the specified parameters for a given field.

        If more than one field is added, the conditions will be joined with a logical and. If that is the case,
        the provided values for `lt` and/or `gt` must match the length of the columns. Otherwise, it will raise
        an error.

        Args:
            column: Field to perform the check in
            lt: Keep only alerts with values less than this
            gt: Keep only alerts with values greater than this
        """
        self.clear_caches()
        mask = pd.Series(True, index=self._alerts.index, dtype=bool)
        if isinstance(column, str):
            column = [column]
        if isinstance(lt, (float, int)):
            lt = [lt]
        if isinstance(gt, (float, int)):
            gt = [gt]
        for i, col in enumerate(column):
            series = self._alerts[col]
            if lt is not None and lt[i] is not None:
                mask &= (series < lt[i])
            if gt is not None and gt[i] is not None:
                mask &= (series > gt[i])
        self._alerts = self._alerts[mask]

    def clear_caches(self):
        """Clears the cache of all methods that use caching"""
        for method in dir(self):
            try:
                getattr(getattr(self, method), "cache_clear")()
            except AttributeError:
                pass

    @abc.abstractmethod
    def _post_process_alerts(self, **kwargs):
        """Adds extra fields to the alert and checks that no unknown kwargs are passed. Should be called with
        `super` in subclass implementations."""
        if "extras" in kwargs:
            self.__add_extra_fields(kwargs.pop("alerts"), kwargs.pop("extras"))
        if set(kwargs) - {"alerts", "surveys", "legacy"}:
            # Only used in subclasses, when using super it should be empty
            raise ValueError(f"Unrecognized kwargs: {', '.join(set(kwargs) - {'alerts', 'surveys', 'legacy'})}")

    def _clear(self, surveys: str | tuple[str, ...], bands: str | tuple[str, ...], extras: list[str]):
        """Keep only alerts in the required surveys and bands and with defined values in required columns.

        The extra fields can have undefined values. These are searched for at the main alert level and inside
        the field `extra_fields`.

        Args:
            surveys: Surveys to keep. Empty tuple to keep all
            bands: Bands to keep. Empty tuple to keep all
            extras: Extra fields to keep
        """
        self.__discard_invalid_alerts(extras)
        self.__discard_not_in_bands(bands)
        self.__discard_not_in_surveys(surveys)

    def __discard_invalid_alerts(self, extras: list[str]):
        """Removes alerts with undefined values in the required fields.

        Will remove all fields not defined in the required fields unless they are listed in `extras`.

        Note that fields in `extras` can have undefined values.

        Args:
            extras: List of additional fields to keep for the alerts
        """
        with pd.option_context("mode.use_inf_as_na", True):
            self._alerts = self._alerts[self._alerts[self.COLUMNS].notna().all(axis="columns")]
        index = (self.INDEX,) if isinstance(self.INDEX, str) else self.INDEX
        self._alerts = self._alerts[[c for c in set(self.COLUMNS + extras) if c not in index]]

    def __discard_not_in_surveys(self, surveys: str | tuple[str]):
        """Keep only alerts in given survey(s). Based on field `sid`.

        Args:
            surveys: Surveys to select. Empty tuple will select all
        """
        self._alerts = self._alerts[self._surveys_mask(surveys)]

    def __discard_not_in_bands(self, bands: str | tuple[str]):
        """Keep only alerts in given band(s). Based on field `fid`.

        Args:
            bands: Bands to select. Empty tuple will select all
        """
        self._alerts = self._alerts[self._bands_mask(bands)]

    def __add_extra_fields(self, alerts: list[dict], extras: list[str]):
        """Include additional fields in the data kept from the alerts.

        These fields can be either at the main alert level or inside the field `extra_fields`. Not all of them
        have to be at the same level, though. For instance, if `extras` is `["field1", "field2"]` then both can be
        directly within the alerts, inside `extra_fields` or one can be in the former, while the other is in
        the latter.

        Args:
            alerts: Original alerts
            extras: List with additional fields
        """
        extras = [extra for extra in extras if extra not in self.COLUMNS and extra not in self._alerts.columns]
        if extras and self._alerts.size:
            records = {alert[self.INDEX]: alert["extra_fields"] for alert in alerts}
            df = pd.DataFrame.from_dict(records, orient="index", columns=extras)
            df = df.reset_index(names=[self.INDEX]).drop_duplicates(self.INDEX).set_index(self.INDEX)
            self._alerts = self._alerts.join(df)
        elif extras:  # Add extra (empty) columns to empty dataframe
            self._alerts[[extras]] = None

    def __mask(self, column: str, values: tuple[str]):
        """Creates a mask for alerts over a string field, based on case-insensitive matching with the given values.

        Args:
            column: Field to make the match(es) against
            values: Values to compare when creating the mask. An empty tuple creates a mask including all alerts

        Returns:
            pd.Series: Boolean mask based on the alerts and the matching of the required columns
        """
        if not values:
            return pd.Series(True, index=self._alerts.index)
        masks = (self._alerts[column].str.lower() == value.lower() for value in values)
        return functools.reduce(lambda l, r: l.__or__(r), masks)

    def _surveys_mask(self, surveys: str | tuple[str, ...] = ()) -> pd.Series:
        """Generate a mask for alerts matching the required surveys (based on field `sid`).

        Args:
            surveys: Surveys to include in the mask. An empty tuple will keep all alerts

        Returns:
            pd.Series: Boolean mask with all required surveys
        """
        surveys = (surveys,) if isinstance(surveys, str) else surveys
        return self.__mask("sid", surveys)

    def _bands_mask(self, bands: str | tuple[str, ...] = ()) -> pd.Series:
        """Generate a mask for alerts matching the required bands (based on field `fid`).

        Args:
            bands: Bands to include in the mask. An empty tuple will keep all alerts

        Returns:
            pd.Series: Boolean mask with all required bands
        """
        bands = (bands,) if isinstance(bands, str) else bands
        return self.__mask("fid", bands)

    @methodtools.lru_cache()
    def get_objects(self) -> np.ndarray:
        """Get array with objects represented among the alerts.

        Returns:
            np.ndarray: Unique objects within alerts
        """
        return self._alerts["id"].unique()

    @methodtools.lru_cache()
    def get_alerts(self, *, surveys: tuple[str, ...] = (), bands: tuple[str, ...] = ()) -> pd.DataFrame:
        """Get a selection of the alerts.

        Args:
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all

        Returns:
            pd.DataFrame: Selected alerts (only columns defined by the class/at initialization are included)
        """
        return self._alerts[self._surveys_mask(surveys) & self._bands_mask(bands)]

    @methodtools.lru_cache()
    def get_grouped(
        self, *, by_fid: bool = False, surveys: tuple[str, ...] = (), bands: tuple[str, ...] = ()
    ) -> DataFrameGroupBy:
        """Get selected detections grouped by object (and, optionally, band).

        Args:
            by_fid: Whether to also group by band
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all

        Returns:
            DataFrameGroupBy: Grouped detections
        """
        return self.get_alerts(surveys=surveys, bands=bands).groupby(["id", "fid"] if by_fid else "id")

    @methodtools.lru_cache()
    def get_which_index(
        self,
        *,
        which: Literal["first", "last"],
        by_fid: bool = False,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
    ) -> pd.Series:
        """Get first or last alert index for every object.

        Args:
            which: Either `first` or `last`
            by_fid: Whether to get the first or last index by band as well
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all

        Returns:
            pd.Series: First or last alert index. Indexed by `id` (and `fid` if `by_fid`).
        """
        if which not in ("first", "last"):
            raise ValueError(f"Unrecognized value for 'which': {which} (can only be first or last)")
        function = "idxmin" if which == "first" else "idxmax"
        return self.get_aggregate("mjd", function, by_fid=by_fid, surveys=surveys, bands=bands)

    def get_which_value(
        self,
        column: str,
        *,
        which: Literal["first", "last"],
        by_fid: bool = False,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
    ) -> pd.Series:
        """Get first or last value of a given field for each object

        Args:
            column: Field for which to select values
            which: Either `first` or `last`
            by_fid: Whether to get the first or last value by band as well
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all

        Returns:
            pd.Series: First or last field value. Indexed by `id` (and `fid` if `by_fid`).
        """
        idx = self.get_which_index(which=which, by_fid=by_fid, surveys=surveys, bands=bands)
        df = self.get_alerts(surveys=surveys, bands=bands)
        return df[column][idx].set_axis(idx.index)

    @methodtools.lru_cache()
    def get_aggregate(
        self,
        column: str,
        func: str | Callable,
        *,
        by_fid: bool = False,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
        **kwargs,
    ) -> pd.Series:
        """Get aggregate value over a given field for each object.

        Args:
            column: Field over which the aggregation is performed
            func: Method name or callable that performs the aggregation per object
            by_fid: Whether to perform the aggregation by band as well
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all
            kwargs: Keyword arguments passed to function

        Returns:
            pd.Series: Aggregate field value. Indexed by `id` (and `fid` if `by_fid`).
        """
        return self.get_grouped(by_fid=by_fid, surveys=surveys, bands=bands)[column].agg(func, **kwargs)

    @methodtools.lru_cache()
    def get_delta(
        self,
        column: str,
        *,
        by_fid: bool = False,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
    ) -> pd.Series:
        """Get difference between maximum and minimum values od a given field for each object.

        Args:
            column: Field to calculate the difference over
            by_fid: Whether to calculate the difference by band as well
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all

        Returns:
            pd.Series: Field value difference. Indexed by `id` (and `fid` if `by_fid`).
        """
        vmax = self.get_aggregate(column, "max", by_fid=by_fid, surveys=surveys, bands=bands)
        vmin = self.get_aggregate(column, "min", by_fid=by_fid, surveys=surveys, bands=bands)
        return vmax - vmin

    def apply(
        self,
        func: Callable[[pd.DataFrame, ...], pd.Series],
        *,
        by_fid: bool = False,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
        **kwargs,
    ) -> pd.DataFrame:
        """Apply given function over the alerts for each object.

        Args:
            func: Function to apply to alerts
            by_fid: Whether to apply the function by band as well
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all
            **kwargs: Keyword arguments passed to `func`

        Returns:
            pd.DataFrame: Results of `func`. Indexed by `id` (and `fid` if `by_fid`).
        """
        return self.get_grouped(by_fid=by_fid, surveys=surveys, bands=bands).apply(func, **kwargs)
