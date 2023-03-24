import warnings

import numpy as np
import pandas as pd

from .base import BaseStrategy


class ZTFStrategy(BaseStrategy):
    """Strategy for ZTF corrections.

    Whether an alert is considered near a source is based on `distnr` being less than `DISTANCE_THRESHOLD`.

    There are 3 criteria for dubiousness:

    * If the object is not near a source but has a negative flux difference (these should be VSs, with a known source)
    * If the first detection for its AID and FID is near a source, but the detection isn't (has it moved?)
    * If the first detection for its AID and FID is not near a source, but the detection is (has it moved?)
    """
    EXTRA_FIELDS = ["magnr", "sigmagnr", "distnr", "distpsnr1", "sgscore1", "chinr", "sharpnr"]
    DISTANCE_THRESHOLD = 1.4
    SCORE_THRESHOLD = 0.4
    CHINR_THRESHOLD = 2
    SHARPNR_MAX = 0.1
    SHARPNR_MIN = -0.13

    @property
    def corrected(self) -> pd.Series:
        return self._extras["distnr"] < self.DISTANCE_THRESHOLD

    @property
    def dubious(self) -> pd.Series:
        negative = self._generic["isdiffpos"] == -1
        return (~self.corrected & negative) | (self._first & ~self.corrected) | (~self._first & self.corrected)

    @property
    def stellar(self) -> pd.Series:
        near_ps1 = self._extras["distpsnr1"] < self.DISTANCE_THRESHOLD
        stellar_ps1 = self._extras["sgscore1"] > self.SCORE_THRESHOLD

        within_sharpnr = (self.SHARPNR_MIN < self._extras["sharpnr"]) < self.SHARPNR_MAX
        stellar_ztf = (self._extras["chinr"] < self.CHINR_THRESHOLD) & within_sharpnr
        return (self.corrected & near_ps1 & stellar_ps1) | (self.corrected & ~near_ps1 & stellar_ztf)

    @property
    def _first(self) -> pd.Series:
        """Whether the first detection for each AID and FID has a nearby known source"""
        corrected = self.corrected[self._generic.groupby(["aid", "fid"])["mjd"].transform("idxmin")]
        corrected.index = self.corrected.index
        return corrected

    def _correct(self) -> pd.DataFrame:
        corrections = super()._correct()

        aux1 = 10 ** (-.4 * self._extras["magnr"].astype(float))
        aux2 = 10 ** (-.4 * self._generic["mag"])
        aux3 = np.maximum(aux1 + self._generic["isdiffpos"] * aux2, 0.0)
        with warnings.catch_warnings():
            # possible log10 of 0; expected and proper result are given
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            corrections["mag_corr"] = -2.5 * np.log10(aux3)

        aux4 = (aux2 * self._generic["e_mag"]) ** 2 - (aux1 * self._extras["sigmagnr"].astype(float)) ** 2
        with warnings.catch_warnings():
            # possible sqrt of negative and division by 0; expected and proper results are given
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            corrections["e_mag_corr"] = np.where(aux4 < 0, np.inf, np.sqrt(aux4) / aux3)
            corrections["e_mag_corr_ext"] = aux2 * self._generic["e_mag"] / aux3

        return corrections
