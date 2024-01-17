import methodtools
import numpy as np
import pandas as pd

from ..utils import functions
from ._base import BaseHandler


class DetectionsHandler(BaseHandler):
    """Class for handling detections.

    Indexed by `candid`.

    Criteria for uniqueness is based on `id` (`oid`, depending on use of `legacy`), `fid` and `mjd`.

    Required fields are: `id`, `sid`, `fid`, `mjd`, `mag` and `e_mag`.

    Additional fields required are `mag_ml` and `e_mag_ml`, but are generated at initialization. The value for these
    fields depend on the argument `corr`. If `False` they will take the values of `mag` and `e_mag`, respectively, for
    all objects. Otherwise, it will check whether the first detection for a given object is corrected. If so, it will
    fill in the values of `mag_corr` and `e_mag_corr_ext`, respectively. If the first detection is not corrected, it
    will fall back to using `mag` and `e_mag`. Note that different objects can use different fields, depending on the
    corrected status of their first detection (when `corr` is `True`).

    Keyword argument:
        corr (bool): Whether to use corrected magnitudes if available. Defaults to `False`
    """

    _NAME = "detections"
    INDEX = "candid"
    NON_DUPLICATE = ["oid", "candid"]
    UNIQUE = ["id", "fid", "mjd"]
    COLUMNS = BaseHandler.COLUMNS + [
        "mag",
        "e_mag",
        "mag_ml",
        "e_mag_ml",
        "forced",
    ]

    def _post_process(self, **kwargs):
        """Handles legacy alerts (renames old field names to the new conventions) and sets the
        `mag_ml` and `e_mag_ml` fields based on the `corr` argument. This is in addition to base
        post-processing"""
        if kwargs.pop("corr", False):
            self.logger.debug(
                "Using corrected magnitudes (if object first detection is corrected)"
            )
            self._use_corrected()
        else:
            self.logger.debug("Using uncorrected magnitudes for all objects")
            self._alerts = self._alerts.assign(
                mag_ml=self._alerts["mag"], e_mag_ml=self._alerts["e_mag"]
            )
        super()._post_process(**kwargs)

    def _use_corrected(self):
        """Sets corrected magnitudes, based on whether the first alert for an object is corrected."""
        idx = self._alerts.groupby("id")["mjd"].idxmin()
        corrected = (
            self._alerts["corrected"][idx]
            .set_axis(idx.index)
            .reindex(self._alerts["id"])
        )

        mag = np.where(
            corrected, self._alerts["mag_corr"], self._alerts["mag"]
        )
        e_mag = np.where(
            corrected, self._alerts["e_mag_corr_ext"], self._alerts["e_mag"]
        )

        self._alerts = self._alerts.assign(mag_ml=mag, e_mag_ml=e_mag)

    @methodtools.lru_cache()
    def get_colors(
        self,
        func: str,
        bands: tuple[str, str],
        *,
        surveys: tuple[str, ...] = (),
        ml: bool = True,
        flux: bool = False,
        **kwargs,
    ) -> pd.Series:
        """Calculate colors (magnitude difference between bands) for all objects.

        Args:
            func: Aggregation function used to compute the colors (e.g., mean, max, etc.)
            bands: Two element tuple with the band names involved in color calculation. Color is first minus second
            surveys: Surveys to select (based on `sid`). Empty tuple selects all
            ml: Whether to use corrected magnitudes (if available)
            flux: Consider magnitudes as flux (computes ratio rather than difference)
            kwargs: Keyword arguments passed to function

        Returns:
            pd.Series: Aggregated color for each object. Indexed by `id`
        """
        first, second = bands

        column = f"mag{'_ml' if ml else ''}"
        mags = self.agg(column, func, by_fid=True, surveys=surveys, **kwargs)
        mags = functions.fill_index(mags, fid=bands)

        if flux:  # + 1 in denominator to avoid division errors
            return mags.xs(first, level="fid") / (
                mags.xs(second, level="fid") + 1
            )
        return mags.xs(first, level="fid") - mags.xs(second, level="fid")
