from __future__ import annotations

import abc
import functools
import logging
from typing import Callable, Literal

import methodtools
import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy


class BaseHandler(abc.ABC):
    """Base class for handling alerts, typically detections (including forced photometry) and non-detections.

    This provides the general base methods for extracting statistics and applying functions to the alerts
    considered.

    Alerts require at the very least fields `sid`, `fid`, `mjd` and `oid` (the latter will be renamed to `id`).
    Usually, a subclass will define additional fields that are required, but missing one of the aforementioned fields
    will result in an initialization error, unless the initialization methods are adapted as well. These always
    required fields are defined the class attribute `COLUMNS`.

    It is possible to define one or more fields that define a unique alert with the class attribute `UNIQUE`.
    Alert duplicates (based on these columns) will be eliminated, keeping only one of these alerts. It is also
    possible to define one or more fields for indexing the alerts using the class attribute `INDEX`. Both unique
    and index fields are optional and should be defined in subclasses.

    Note that there is heavy use of caching, so externally modifying the alerts within the class can cause wrong
    results to come up. For development, it is suggested that methods that can modify the internal data frame
    make use of the method `clear_caches` to ensure that subsequent calls on the main methods give consistent results.
    """

    _NAME: str  # Only used in logging
    INDEX: str | list[str] = []
    UNIQUE: str | list[str] = []
    NON_DUPLICATE: str | list[str] = []
    COLUMNS = ["id", "sid", "fid", "mjd"]

    def __init__(
        self,
        alerts: list[dict],
        *,
        surveys: str | tuple[str] = (),
        bands: str | tuple[str] = (),
        **kwargs,
    ):
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
            extras (list[str]): Additional fields to keep (these fields can have undefined values)
        """
        self.logger = logging.getLogger(f"alerce.{self.__class__.__name__}")
        self.logger.info(f"Creating {self.__class__.__name__}")
        try:
            self._alerts = pd.DataFrame.from_records(
                alerts, exclude=["extra_fields"]
            )
        except KeyError:  # extra_fields is not present
            self._alerts = pd.DataFrame.from_records(alerts)
        self._alerts = self._alerts.rename(columns={"oid": "id"})
        if self._alerts.size == 0:
            index = (
                {self.INDEX}
                if isinstance(self.INDEX, str)
                else set(self.INDEX)
            )
            unique = (
                {self.UNIQUE}
                if isinstance(self.UNIQUE, str)
                else set(self.UNIQUE)
            )

            columns = set(self.COLUMNS) | index | unique
            self._alerts = pd.DataFrame(columns=list(columns))
        self.logger.info(
            f"Total {self._NAME} before clearing: {len(self._alerts)}"
        )

        if self.UNIQUE:
            self._alerts.drop_duplicates(self.UNIQUE, inplace=True)
            self.logger.debug(
                f"{len(self._alerts)} {self._NAME} remain after unque removal"
            )
        if self.NON_DUPLICATE:
            self._alerts.drop_duplicates(self.NON_DUPLICATE, inplace=True)
            self.logger.debug(
                f"{len(self._alerts)} {self._NAME} remain after duplicate removal"
            )
        if self.INDEX:
            self._alerts.set_index(self.INDEX, inplace=True)
            self.logger.debug(f"Using column(s) {self.INDEX} for indexing")

        extras = kwargs.get("extras", [])
        self._post_process(alerts=alerts, surveys=surveys, **kwargs)
        self._clear(surveys=surveys, bands=bands, extras=extras)
        self.logger.info(
            f"Total {self._NAME} after clearing: {len(self._alerts)}"
        )
        self._alerts = self._alerts.sort_values(["id", "mjd"])

    def not_enough(self, minimum: int, *, by_fid: bool = False):
        """Remove all alerts from objects without a minimum number of detections.

        Args:
            minimum: Minimum number of detections
            by_fid: If `True`, checks that at least one band has the specified minimum

        """
        if minimum <= 1:
            return
        self.clear_caches()
        if by_fid:
            self.logger.debug(
                f"Selecting objects with more than {minimum} {self._NAME} in total"
            )
            mask = (
                self._alerts.groupby("id")["fid"]
                .value_counts()
                .unstack("fid")
                .max(axis="columns")
                > minimum
            )
        else:
            self.logger.debug(
                f"Selecting objects with more than {minimum} {self._NAME} in at least one band"
            )
            mask = self._alerts.groupby("id")["fid"].count() > minimum
        self._alerts = self._alerts[self._alerts["id"].isin(mask.index[mask])]
        self.logger.debug(f"{self.ids().size} objects remain after selection")

    def match(self, other: BaseHandler):
        """Removes alerts from objects contained in the current handler that are not present in the other.

        Only modifies the current handler, the other one will not be modified.

        Args:
            other: Handler used to check for objects
        """
        self.clear_caches()
        self.logger.debug(
            f"Selecting objects present in {other.__class__.__name__} instance"
        )
        self._alerts = self._alerts[self._alerts["id"].isin(other.ids())]
        self.logger.debug(f"{self.ids().size} objects remain after selection")

    def select(
        self,
        column: str | list[str],
        *,
        lt: float | list[float, None] = None,
        gt: float | list[float, None] = None,
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
                mask &= series < lt[i]
            if gt is not None and gt[i] is not None:
                mask &= series > gt[i]
        self._alerts = self._alerts[mask]

    def clear_caches(self):
        """Clears the cache of all methods that use caching"""
        for method in dir(self):
            try:
                getattr(getattr(self, method), "cache_clear")()
            except AttributeError:
                pass

    @abc.abstractmethod
    def _post_process(self, **kwargs):
        """Adds extra fields to the alert and checks that no unknown kwargs are passed. Should be called with
        `super` in subclass implementations."""
        if "extras" in kwargs:
            self.__add_extra_fields(kwargs.pop("alerts"), kwargs.pop("extras"))
        if set(kwargs) - {"alerts", "surveys"}:
            # Only used in subclasses, when using super it should be empty
            raise ValueError(
                f"Unrecognized kwargs: {', '.join(set(kwargs) - {'alerts', 'surveys'})}"
            )

    def _clear(
        self,
        surveys: str | tuple[str, ...],
        bands: str | tuple[str, ...],
        extras: list[str],
    ):
        """Keep only alerts in the required surveys and bands and with defined values in required columns.

        The extra fields can have undefined values. These are searched for at the main alert level and inside
        the field `extra_fields`.

        Args:
            surveys: Surveys to keep. Empty tuple to keep all
            bands: Bands to keep. Empty tuple to keep all
            extras: Extra fields to keep
        """
        self.__discard_invalid_alerts(extras)
        self.__discard_not_in_surveys(surveys)
        self.__discard_not_in_bands(bands)

    def __discard_invalid_alerts(self, extras: list[str]):
        """Removes alerts with undefined values in the required fields.

        Will remove all fields not defined in the required fields unless they are listed in `extras`.

        Note that fields in `extras` can have undefined values.

        Args:
            extras: List of additional fields to keep for the alerts
        """
        self.logger.debug(
            f"Selecting valid {self._NAME} (i.e., no undefined values in {self.COLUMNS})"
        )
        with pd.option_context("mode.use_inf_as_na", True):
            self._alerts = self._alerts[
                self._alerts[self.COLUMNS].notna().all(axis="columns")
            ]
        self.logger.debug(
            f"{len(self._alerts)} {self._NAME} remain after selection"
        )

        index = (self.INDEX,) if isinstance(self.INDEX, str) else self.INDEX
        self._alerts = self._alerts[
            [c for c in set(self.COLUMNS + extras) if c not in index]
        ]

    def __discard_not_in_surveys(self, surveys: str | tuple[str]):
        """Keep only alerts in given survey(s). Based on field `sid`.

        Args:
            surveys: Surveys to select. Empty tuple will select all
        """
        self.logger.debug(
            f"Selecting {self._NAME} from survey(s): {surveys if surveys else 'all'}"
        )
        self._alerts = self._alerts[self._surveys_mask(surveys)]
        self.logger.debug(
            f"{len(self._alerts)} {self._NAME} remain after selection"
        )

    def __discard_not_in_bands(self, bands: str | tuple[str]):
        """Keep only alerts in given band(s). Based on field `fid`.

        Args:
            bands: Bands to select. Empty tuple will select all
        """
        self.logger.debug(
            f"Selecting {self._NAME} from band(s): {bands if bands else 'all'}"
        )
        self._alerts = self._alerts[self._bands_mask(bands)]
        self.logger.debug(
            f"{len(self._alerts)} {self._NAME} remain after selection"
        )

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
        extras = [
            extra
            for extra in extras
            if extra not in self.COLUMNS and extra not in self._alerts.columns
        ]
        if extras and self._alerts.size:
            records = {
                alert[self.INDEX]: {**alert["extra_fields"], "id": alert["oid"]} for alert in alerts
            }
            df = pd.DataFrame.from_dict(
                records, orient="index", columns=extras + ["id"]
            )
            df = (
                df.reset_index(names=[self.INDEX])
                .drop_duplicates(self.NON_DUPLICATE)
                .set_index(self.INDEX)
            )
            self._alerts = self._alerts.join(df, how="left", rsuffix="_extra").drop("id_extra", axis=1)
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
        masks = (
            self._alerts[column].str.lower() == value.lower()
            for value in values
        )
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

    def add_field(self, name: str, values: pd.Series | np.ndarray | float):
        """Add a new, arbitrary field to the alerts.

        Args:
            name: Field name. It will override any existing field with the same name
            values: Values for the new field. It is responsibility of the user to make sure it matches with alerts
        """
        self.clear_caches()
        self._alerts = self._alerts.assign(**{name: values})

    @methodtools.lru_cache()
    def ids(self) -> np.ndarray:
        """Get array with object IDs represented among the alerts.

        Returns:
            np.ndarray: Unique object IDs within alerts
        """
        return self._alerts["id"].unique()

    @methodtools.lru_cache()
    def alerts(
        self,
        *,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
        flag: str = None,
    ) -> pd.DataFrame:
        """Get a selection of the alerts.

        Args:
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all
            flag: Field used as flag to select specific alerts (assumes it is boolean)

        Returns:
            pd.DataFrame: Selected alerts (only columns defined by the class/at initialization are included)
        """
        mask = self._surveys_mask(surveys) & self._bands_mask(bands)
        if flag is not None:
            mask &= self._alerts[flag]
        return self._alerts[mask]

    @methodtools.lru_cache()
    def grouped(
        self,
        *,
        by_fid: bool = False,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
        flag: str = None,
    ) -> DataFrameGroupBy:
        """Get selected detections grouped by object (and, optionally, band).

        Args:
            by_fid: Whether to also group by band
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all
            flag: Field used as flag to select specific alerts (assumes it is boolean)

        Returns:
            DataFrameGroupBy: Grouped detections
        """
        return self.alerts(surveys=surveys, bands=bands, flag=flag).groupby(
            ["id", "fid"] if by_fid else "id"
        )

    @methodtools.lru_cache()
    def which_index(
        self,
        *,
        which: Literal["first", "last"],
        by_fid: bool = False,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
        flag: str = None,
    ) -> pd.Series:
        """Get first or last alert index for every object.

        Args:
            which: Either `first` or `last`
            by_fid: Whether to get the first or last index by band as well
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all
            flag: Field used as flag to select specific alerts (assumes it is boolean)

        Returns:
            pd.Series: First or last alert index. Indexed by `id` (and `fid` if `by_fid`).
        """
        if which not in ("first", "last"):
            raise ValueError(
                f"Unrecognized value for 'which': {which} (can only be first or last)"
            )
        function = "idxmin" if which == "first" else "idxmax"
        return self.agg(
            "mjd",
            function,
            by_fid=by_fid,
            surveys=surveys,
            bands=bands,
            flag=flag,
        )

    def which_value(
        self,
        column: str,
        *,
        which: Literal["first", "last"],
        by_fid: bool = False,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
        flag: str = None,
    ) -> pd.Series:
        """Get first or last value of a given field for each object

        Args:
            column: Field for which to select values
            which: Either `first` or `last`
            by_fid: Whether to get the first or last value by band as well
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all
            flag: Field used as flag to select specific alerts (assumes it is boolean)

        Returns:
            pd.Series: First or last field value. Indexed by `id` (and `fid` if `by_fid`).
        """
        idx = self.which_index(
            which=which, by_fid=by_fid, surveys=surveys, bands=bands, flag=flag
        )
        df = self.alerts(surveys=surveys, bands=bands, flag=flag)
        return df[column][idx].set_axis(idx.index)

    @methodtools.lru_cache()
    def agg(
        self,
        column: str,
        func: str | Callable,
        *,
        by_fid: bool = False,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
        flag: str = None,
        **kwargs,
    ) -> pd.Series:
        """Get aggregate value over a given field for each object.

        Args:
            column: Field over which the aggregation is performed
            func: Method name or callable that performs the aggregation per object
            by_fid: Whether to perform the aggregation by band as well
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all
            flag: Field used as flag to select specific alerts (assumes it is boolean)
            kwargs: Keyword arguments passed to function

        Returns:
            pd.Series: Aggregate field value. Indexed by `id` (and `fid` if `by_fid`).
        """
        return self.grouped(
            by_fid=by_fid, surveys=surveys, bands=bands, flag=flag
        )[column].agg(func, **kwargs)

    @methodtools.lru_cache()
    def get_delta(
        self,
        column: str,
        *,
        by_fid: bool = False,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
        flag: str = None,
    ) -> pd.Series:
        """Get difference between maximum and minimum values od a given field for each object.

        Args:
            column: Field to calculate the difference over
            by_fid: Whether to calculate the difference by band as well
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all
            flag: Field used as flag to select specific alerts (assumes it is boolean)

        Returns:
            pd.Series: Field value difference. Indexed by `id` (and `fid` if `by_fid`).
        """
        vmax = self.agg(
            column,
            "max",
            by_fid=by_fid,
            surveys=surveys,
            bands=bands,
            flag=flag,
        )
        vmin = self.agg(
            column,
            "min",
            by_fid=by_fid,
            surveys=surveys,
            bands=bands,
            flag=flag,
        )
        return vmax - vmin

    def apply(
        self,
        func: Callable[[pd.DataFrame, ...], pd.Series],
        *,
        by_fid: bool = False,
        surveys: tuple[str, ...] = (),
        bands: tuple[str, ...] = (),
        flag: str = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Apply given function over the alerts for each object.

        Args:
            func: Function to apply to alerts
            by_fid: Whether to apply the function by band as well
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            bands: Bands to select (based on `fid`). Empty tuple selects all
            flag: Field used as flag to select specific alerts (assumes it is boolean)
            **kwargs: Keyword arguments passed to `func`

        Returns:
            pd.DataFrame: Results of `func`. Indexed by `id` (and `fid` if `by_fid`).
        """
        return self.grouped(
            by_fid=by_fid, surveys=surveys, bands=bands, flag=flag
        ).apply(func, **kwargs)
