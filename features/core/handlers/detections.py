import numpy as np
import pandas as pd

from ..utils import fill_index
from ._base import BaseHandler


class DetectionsHandler(BaseHandler):
    INDEX = "candid"
    UNIQUE = "candid"
    _NON_NULL_COLUMNS = BaseHandler._NON_NULL_COLUMNS + ["mag_ml", "e_mag_ml", "isdiffpos", "ra", "dec", "corrected"]

    def _post_process_alerts(self, **kwargs):
        if "extras" in kwargs:
            extras = self.__extract_extra_fields(kwargs.pop("alerts"), kwargs.pop("extras"))
            self._alerts = self._alerts.join(extras)
        self._machine_learning_magnitudes(kwargs.pop("surveys"), kwargs.pop("corrected", False))
        super()._post_process_alerts(**kwargs)

    def _machine_learning_magnitudes(self, surveys: tuple[str, ...], corrected: bool):
        if not corrected:
            self._alerts = self._alerts.assign(mag_ml=self._alerts["mag"], e_mag_ml=self._alerts["e_mag"])

        corrected = self.get_which_value("corrected", which="first", surveys=surveys)
        corrected = corrected.reindex(self._alerts["aid"])

        mag = np.where(corrected, self._alerts["mag_corr"], self._alerts["mag"])
        e_mag = np.where(corrected, self._alerts["e_mag_corr_ext"], self._alerts["e_mag"])

        self._alerts = self._alerts.assign(mag_ml=mag, e_mag_ml=e_mag)

    @classmethod
    def __extract_extra_fields(cls, alerts: list[dict], extras: list[str]) -> pd.DataFrame:
        df = {alert[cls.INDEX]: alert["extra_fields"] for alert in alerts}
        df = pd.DataFrame.from_dict(df, orient="index", columns=extras)
        return df.reset_index(names=[cls.INDEX]).drop_duplicates(cls.INDEX).set_index(cls.INDEX)

    def get_colors(
        self, func: str, bands: tuple[str, str], *, surveys: str | tuple[str, ...] = (), ml: bool = False
    ) -> pd.Series:
        first, second = bands

        mags = self.get_aggregate(f"mag{'_ml' if ml else ''}", func, by_fid=True, surveys=surveys, bands=bands)
        mags = fill_index(mags, fid=bands)
        return mags.xs(first, level="fid") - mags.xs(second, level="fid")

    def get_count_by_sign(self, sign: int, *, by_fid: bool = False, bands: str | tuple[str, ...] = ()) -> pd.Series:
        counts = self.get_grouped(by_fid=by_fid, bands=bands)["isdiffpos"].value_counts()
        if by_fid and bands:
            return fill_index(counts, fill_value=0, fid=bands, isdiffpos=(-1, 1)).xs(sign, level="isdiffpos")
        return fill_index(counts, fill_value=0, isdiffpos=(-1, 1)).xs(sign, level="isdiffpos")
