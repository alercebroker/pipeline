import methodtools
import numpy as np
import pandas as pd

from ..utils import functions
from ._base import BaseHandler


class DetectionsHandler(BaseHandler):
    """Class for handling detections.

    Indexed by `candid`.

    Criteria for uniqueness is based on `id` (`aid` or `oid`, depending on use of `legacy`), `fid` and `mjd`.

    Required fields are `id`, `sid`, `fid`, `mjd`, `mag`, `e_mag` and `isdiffpos`.

    Additional fields required are `mag_ml` and `e_mag_ml`, but are generated at initialization.

    Additional keyword argument:
        corr (bool): Whether to use corrected magnitudes if available. Defaults to `False`

    The value for fields `mag_ml` and `e_mag_ml` depend on the argument `corr`. If `False` they will take the values
    of `mag` and `e_mag`, respectively, for all objects. Otherwise, it will check whether the first object detection
    is corrected. If so, it will fill in the values of `mag_corr` and `e_mag_corr_ext`, respectively. If the first
    detection is not corrected, it will fall back to using `mag` and `e_mag`. Note that different objects can use
    different values, depending on the corrected status of their first detection if `corr` is `True`.
    """

    INDEX = "candid"
    UNIQUE = ["id", "fid", "mjd"]
    COLUMNS = BaseHandler.COLUMNS + ["mag", "e_mag", "mag_ml", "e_mag_ml", "isdiffpos"]

    def _post_process_alerts(self, **kwargs):
        """Handles legacy alerts (renames old field names to the new conventions) and sets the
        `mag_ml` and `e_mag_ml` fields based on the `corr` argument. This is in addition to base
        post-processing"""

        if kwargs.pop("legacy", False):
            self._alerts["mag"] = self._alerts["magpsf"]
            self._alerts["e_mag"] = self._alerts["sigmapsf"]
            self._alerts["mag_corr"] = self._alerts["magpsf_corr"]
            self._alerts["e_mag_corr"] = self._alerts["sigmapsf_corr"]
            self._alerts["e_mag_corr_ext"] = self._alerts["sigmapsf_corr_ext"]
        if kwargs.pop("corr", False):
            self._use_corrected_magnitudes(kwargs.pop("surveys"))
        else:
            self._alerts = self._alerts.assign(mag_ml=self._alerts["mag"], e_mag_ml=self._alerts["e_mag"])
        super()._post_process_alerts(**kwargs)

    def _use_corrected_magnitudes(self, surveys: tuple[str, ...]):
        """Sets corrected magnitudes, based on whether the first alert for an object is corrected.

        Args:
            surveys: Surveys that need correction checking
        """
        idx = self._alerts[self._surveys_mask(surveys)].groupby("id")["mjd"].idxmin()
        corrected = self._alerts["corrected"][idx].set_axis(idx.index).reindex(self._alerts["id"])

        mag = np.where(corrected, self._alerts["mag_corr"], self._alerts["mag"])
        e_mag = np.where(corrected, self._alerts["e_mag_corr_ext"], self._alerts["e_mag"])

        self._alerts = self._alerts.assign(mag_ml=mag, e_mag_ml=e_mag)

    @methodtools.lru_cache()
    def get_colors(
        self, func: str, bands: tuple[str, str], *, surveys: tuple[str, ...] = (), ml: bool = True
    ) -> pd.Series:
        """Calculate colors (magnitude difference between bands) for all objects.

        Args:
            func: Aggregation function used to compute the colors (e.g., mean, max, etc.)
            bands: Two element tuple with the band names involved in color calculation. Color is first minus second
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            ml: Whether to use corrected magnitudes (if available)

        Returns:
            pd.Series: Aggregated color for each object. Indexed by `id`
        """
        first, second = bands

        mags = self.get_aggregate(f"mag{'_ml' if ml else ''}", func, by_fid=True, surveys=surveys, bands=bands)
        mags = functions.fill_index(mags, fid=bands)
        return mags.xs(first, level="fid") - mags.xs(second, level="fid")

    @methodtools.lru_cache()
    def get_count_by_sign(self, sign: int, *, by_fid: bool = False, bands: tuple[str, ...] = ()) -> pd.Series:
        """Number of detections with a given sign

        Args:
            sign: Either 1 (counts positive difference detections) or -1 (counts negative difference detections)
            by_fid: Whether to count detections by band as well
            bands: Bands to select (based on `fid`). Empty tuple selects all

        Returns:
            pd.Series: Number detections with given sign. Indexed by `id` (and `fid` if `by_fid`
        """
        counts = self.get_aggregate("isdiffpos", "value_counts", by_fid=by_fid, bands=bands)
        kwargs = dict(isdiffpos=(-1, 1))
        if by_fid and bands:
            kwargs.update(fid=bands)
        return functions.fill_index(counts, fill_value=0, dtype=int, **kwargs).xs(sign, level="isdiffpos")
