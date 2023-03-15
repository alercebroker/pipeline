import warnings

import numpy as np
import pandas as pd

from .base import BaseStrategy


class ZTFStrategy(BaseStrategy):
    EXTRA_FIELDS = ["magnr", "sigmagnr", "distnr"]
    DISTANCE_THRESHOLD = 1.4
    ZERO_MAG = 100.

    @property
    def corrected(self) -> pd.Series:
        return self._extra["distnr"] < self.DISTANCE_THRESHOLD

    @property
    def dubious(self) -> pd.Series:
        negative = ~self.corrected & (self._generic["isdiffpos"] == -1)
        return (~self.corrected & negative) | (self.first & ~self.corrected) | (~self.first & self.corrected)

    @property
    def first(self) -> pd.Series:
        full = self._generic.assign(corrected=self.corrected)
        return full.groupby(["aid", "fid"])["corrected"].transform(min, "mjd")

    def _correct(self) -> pd.DataFrame:
        corrections = super()._correct()

        aux1 = 10 ** (-.4 * self._extra["magnr"])
        aux2 = 10 ** (-.4 * self._generic["mag"])
        aux3 = np.maximum(aux1 + self._generic["isdiffpos"] * aux2, 0.0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            corrections["mag_corr"] = -2.5 * np.log10(aux3)

        aux4 = aux2 ** 2 * self._generic["e_mag"] ** 2 - aux1 ** 2 * self._extra["sigmagnr"] ** 2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            corrections["e_mag_corr"] = np.sqrt(aux4) / aux3
            corrections["e_mag_corr_ext"] = aux2 * self._generic["e_mag"] / aux3

        bad = (self._extra["magnr"] < 0) & (self._generic["mag"] < 0)

        corrections[bad | self.corrected] = np.nan

        corrections.replace(np.inf, self.ZERO_MAG)
        return corrections
