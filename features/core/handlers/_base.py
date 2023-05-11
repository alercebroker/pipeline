from __future__ import annotations

import abc
import functools
from typing import Callable, Literal

import methodtools
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy


class BaseHandler(abc.ABC):
    """Class for handling alerts, typically detections (including forced photometry) and non-detections.

    Here we will use "alert" to refer to any type that is further implemented by subclasses.

    This handler can receive a list of alerts from multiple objects simultaneously and will produce results for each
    object (and band, depending on the options provided when calling the specific method).

    This provides the general base methods for extracting statistics and applying functions to the alert
    set considered. It requires a list of dictionaries as input, with the alert fields used by the ALeRCE pipeline.
    These fields can be found in the input schema for the feature step. Alternatively, old ALeRCE pipeline alerts can
    be used by initializing with the option `legacy` set to `True`.

    Alerts require at the very least fields `sid`, `fid`, `mjd` and `aid` (the last can be `oid` if using `legacy`).
    Usually, a subclass will define additional fields that are required, but missing one of the mentioned above
    will result in an initialization error, unless the initialization methods are adapted as well. These always
    required fields are defined the class attribute `_COLUMNS`.

    It is possible to define one or more fields that define a unique alert with the class attribute `UNIQUE`.
    Alert duplicates (based on these columns) will be eliminated, keeping only one of these alerts. It is also
    possible to define one or more fields for indexing the alerts using the class attribute `INDEX`. Both unique
    and index fields are optional and should be defined in subclasses.

    At initialization, it is possible to select only alerts from certain surveys or bands. These can be passed as
    strings or tuples of strings if more than one survey/band is selected.

    To keep fields besides those defined in `_COLUMNS`, at initialization the option `extras` can be passed as a
    list of strings. These are additional fields to be kept, if present directly in the alert, or taken from inside
    the `extra_fields` field of the alert if present. If not found, these extra columns will be filled with NaNs.

    Note that there is heavy use of caching, so externally modifying the alerts within the class can cause wrong
    results to come up. For development, it is suggested that methods that can modify the internal data frame
    make use of the method `clear_caches` to ensure that subsequent calls on the main methods give consistent results.
    """

    INDEX: str | list[str] = []
    UNIQUE: str | list[str] = []
    _COLUMNS = ["id", "sid", "fid", "mjd"]

    def __init__(self, alerts: list[dict], *, surveys: str | tuple[str] = (), bands: str | tuple[str] = (), **kwargs):
        legacy = kwargs.pop("legacy", False)
        try:
            self._alerts = pd.DataFrame.from_records(alerts, exclude=["extra_fields"])
        except KeyError:  # extra_fields is not present
            self._alerts = pd.DataFrame.from_records(alerts)
        self._alerts = self._alerts.rename(columns={"oid" if legacy else "aid": "id"})
        if self._alerts.size == 0:
            index = {self.INDEX} if isinstance(self.INDEX, str) else set(self.INDEX)
            unique = {self.UNIQUE} if isinstance(self.UNIQUE, str) else set(self.UNIQUE)

            columns = set(self._COLUMNS) | index | unique
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

    def select(self, column: str, *, lt: float = None, gt: float = None, le: bool = False, ge: bool = False):
        """Removes all alerts outside the specified parameters for a given field.

        Args:
            column: Field to perform the check in
            lt: Keep only alerts with values less than this
            gt: Keep only alerts with values greater than this
            le: Compare the value for `lt` using less than equal rather than strictly less
            ge: Compare the value for `gt` using greater than equal rather than strictly greater
        """
        self.clear_caches()
        mask = pd.Series(True, index=self._alerts.index, dtype=bool)
        series = self._alerts[column]
        if lt is not None:
            mask &= (series <= lt) if le else (series < lt)
        if gt is not None:
            mask &= (series >= gt) if ge else (series > gt)
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
        """Adds extra fields to the alert and checks that no unknown kwargs are passed."""
        if "extras" in kwargs:
            self.__add_extra_fields(kwargs.pop("alerts"), kwargs.pop("extras"))
        if set(kwargs) - {"alerts", "surveys", "legacy"}:
            # Only used in subclasses, when using super it should be empty
            raise ValueError(f"Unrecognized kwargs: {', '.join(set(kwargs) - {'alerts', 'surveys', 'legacy'})}")

    def _clear(self, surveys: str | tuple[str, ...], bands: str | tuple[str, ...], extras: list[str]):
        """Remove alerts with NA values in required fields and not in the required surveys and bands.

        Args:
            surveys: Surveys to keep
            bands: Bands to keep
            extras: Extra fields to keeps (not the same defined in `_COLUMNS`)
        """
        self.__discard_invalid_alerts(extras)
        self.__discard_not_in_bands(bands)
        self.__discard_not_in_surveys(surveys)

    def __discard_invalid_alerts(self, extras: list[str]):
        """Removes alerts with NA values in the required `_COLUMNS` and keep only those and `extras`.

        Note that `extras` columns can have NA values.

        Args:
            extras: List of additional fields to keep for the alerts
        """
        with pd.option_context("mode.use_inf_as_na", True):
            self._alerts = self._alerts[self._alerts[self._COLUMNS].notna().all(axis="columns")]
        index = (self.INDEX,) if isinstance(self.INDEX, str) else self.INDEX
        self._alerts = self._alerts[[c for c in self._COLUMNS + extras if c not in index]]

    def __discard_not_in_surveys(self, surveys: str | tuple[str]):
        self._alerts = self._alerts[self._surveys_mask(surveys)]

    def __discard_not_in_bands(self, bands: str | tuple[str]):
        self._alerts = self._alerts[self._bands_mask(bands)]

    def __add_extra_fields(self, alerts: list[dict], extras: list[str]):
        extras = [_ for _ in extras if _ not in self._alerts.columns]
        if extras and self._alerts.size:
            records = {alert[self.INDEX]: alert["extra_fields"] for alert in alerts}
            df = pd.DataFrame.from_dict(records, orient="index", columns=extras)
            df = df.reset_index(names=[self.INDEX]).drop_duplicates(self.INDEX).set_index(self.INDEX)
            self._alerts = self._alerts.join(df)
        elif extras:  # Add extra (empty) columns to empty dataframe
            self._alerts[[extras]] = None

    def __mask(self, column: str, values: tuple[str]):
        if not values:
            return pd.Series(True, index=self._alerts.index)
        masks = (self._alerts[column].str.lower() == value.lower() for value in values)
        return functools.reduce(lambda l, r: l.__or__(r), masks)

    def _surveys_mask(self, surveys: str | tuple[str, ...] = ()) -> pd.Series:
        surveys = (surveys,) if isinstance(surveys, str) else surveys
        return self.__mask("sid", surveys)

    def _bands_mask(self, bands: str | tuple[str, ...] = ()) -> pd.Series:
        bands = (bands,) if isinstance(bands, str) else bands
        return self.__mask("fid", bands)

    @methodtools.lru_cache()
    def get_objects(self):
        return self._alerts["id"].unique()

    @methodtools.lru_cache()
    def get_alerts(self, *, surveys: tuple[str, ...] = (), bands: tuple[str, ...] = ()) -> pd.DataFrame:
        return self._alerts[self._surveys_mask(surveys) & self._bands_mask(bands)]

    @methodtools.lru_cache()
    def get_grouped(
        self, *, by_fid: bool = False, surveys: tuple[str, ...] = (), bands: tuple[str, ...] = ()
    ) -> DataFrameGroupBy:
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
    ) -> pd.Series:
        return self.get_grouped(by_fid=by_fid, surveys=surveys, bands=bands)[column].agg(func)

    @methodtools.lru_cache()
    def get_delta(
        self,
        column: str,
        *,
        by_fid: bool = False,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
    ) -> pd.Series:
        vmax = self.get_aggregate(column, "max", by_fid=by_fid, surveys=surveys, bands=bands)
        vmin = self.get_aggregate(column, "min", by_fid=by_fid, surveys=surveys, bands=bands)
        return vmax - vmin

    def apply_grouped(
        self,
        func: Callable[[pd.DataFrame, ...], pd.Series],
        *,
        by_fid: bool = False,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
        **kwargs,
    ) -> pd.DataFrame:
        return self.get_grouped(by_fid=by_fid, surveys=surveys, bands=bands).apply(func, **kwargs)
