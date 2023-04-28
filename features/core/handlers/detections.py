import methodtools
import numpy as np
import pandas as pd

from ..utils import fill_index
from ._base import BaseHandler


class DetectionsHandler(BaseHandler):
    INDEX = "candid"
    UNIQUE = ["aid", "mjd"]
    _COLUMNS = BaseHandler._COLUMNS + ["mag", "e_mag", "mag_ml", "e_mag_ml", "isdiffpos", "ra", "dec"]

    def _post_process_alerts(self, **kwargs):
        if "extras" in kwargs:
            self.__add_extra_fields(kwargs.pop("alerts"), kwargs.pop("extras"))
        if kwargs.pop("corr", False):
            self._use_corrected_magnitudes(kwargs.pop("surveys"))
        else:
            self._alerts = self._alerts.assign(mag_ml=self._alerts["mag"], e_mag_ml=self._alerts["e_mag"])
        super()._post_process_alerts(**kwargs)

    def _use_corrected_magnitudes(self, surveys: tuple[str, ...]):
        idx = self._alerts[self._surveys_mask(surveys)].groupby("aid")["mjd"].idxmin()
        corrected = self._alerts["corrected"][idx].set_axis(idx.index).reindex(self._alerts["aid"])

        mag = np.where(corrected, self._alerts["mag_corr"], self._alerts["mag"])
        e_mag = np.where(corrected, self._alerts["e_mag_corr_ext"], self._alerts["e_mag"])

        self._alerts = self._alerts.assign(mag_ml=mag, e_mag_ml=e_mag)

    def __add_extra_fields(self, alerts: list[dict], extras: list[str]):
        extras = [_ for _ in extras if _ not in self._alerts.columns]
        if extras and self._alerts.size:
            records = {alert[self.INDEX]: alert["extra_fields"] for alert in alerts}
            df = pd.DataFrame.from_dict(records, orient="index", columns=extras)
            df = df.reset_index(names=[self.INDEX]).drop_duplicates(self.INDEX).set_index(self.INDEX)
            self._alerts = self._alerts.join(df)
        elif extras:  # Add extra (empty) columns to empty dataframe
            self._alerts[[extras]] = None

    @methodtools.lru_cache()
    def get_colors(
        self, func: str, bands: tuple[str, str], *, surveys: str | tuple[str, ...] = (), ml: bool = True
    ) -> pd.Series:
        first, second = bands

        mags = self.get_aggregate(f"mag{'_ml' if ml else ''}", func, by_fid=True, surveys=surveys, bands=bands)
        mags = fill_index(mags, fid=bands)
        return mags.xs(first, level="fid") - mags.xs(second, level="fid")

    @methodtools.lru_cache()
    def get_count_by_sign(self, sign: int, *, by_fid: bool = False, bands: str | tuple[str, ...] = ()) -> pd.Series:
        counts = self.get_grouped(by_fid=by_fid, bands=bands)["isdiffpos"].value_counts()
        if by_fid and bands:
            return fill_index(counts, fill_value=0, fid=bands, isdiffpos=(-1, 1)).xs(sign, level="isdiffpos")
        return fill_index(counts, fill_value=0, isdiffpos=(-1, 1)).xs(sign, level="isdiffpos")
