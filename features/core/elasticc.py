import pandas as pd

from ._base import BaseFeatureExtractor
from .utils import decorators, functions, extras
from .utils.extras.spm import fit_spm_v1


class ELAsTiCCClassifierFeatureExtractor(BaseFeatureExtractor):
    """Extractor for ELAsTiCC light-curve classifier.

    Uses only alerts from LSST.

    Uses corrected magnitudes when possible. Otherwise, defaults are the same as the base extractor.

    The extractors included (and computed) here by default are:

    * `periods`: Period related features (multi-band period, per band period, etc.). Check notes.
    * `fats`: Computes all FATS features defined in `FATS_FEATURES`.
    * `mhps`: Mexican hat power spectrum for each object, per band.
    * `gp_drw`: Gaussian process damp random walk parameters for each object and band.
    * `iqr`: Inter-quartile range in magnitude distribution of each object, per band.
    * `mwebv`: Median Milky Way extinction for each object
    * `redshift_helio`: Median heliocentric redshift for each object
    * `colors`: Compute P90 colors for each object

    Notes:
        If `periods` is excluded, `COMPUTE_KIM`, `N_HARMONICS` and `POWER_RATE_FACTORS` will all be ignored, as they
        depend on values computed in the main period calculations.
    """

    _AUTO_EXCLUDE = {"galactic_coordinates"}
    NAME = "elasticc_lc_features"
    VERSION = "1.0"
    SURVEYS = ("LSST",)
    BANDS = ("u", "g", "r", "i", "z", "Y")
    EXTRA_COLUMNS = ["mwebv", "z_final"]
    USE_CORRECTED = True

    def _discard_detections(self):
        """Exclude noisy detections"""
        self.detections.select(["mag_ml", "e_mag_ml"], lt=[50e3, 300])
        super()._discard_detections()

    @decorators.add_fid(0)
    def calculate_mwebv(self) -> pd.DataFrame:
        return pd.DataFrame({"mwebv": self.detections.get_aggregate("mwebv", "median")})

    @decorators.add_fid(0)
    def calculate_redshift_helio(self) -> pd.DataFrame:
        return pd.DataFrame({"redshift_helio": self.detections.get_aggregate("z_final", "median")})

    @decorators.add_fid("".join(BANDS))
    def calculate_colors(self) -> pd.DataFrame:
        colors = {}
        for b1, b2 in zip(self.BANDS[:-1], self.BANDS[1:]):
            colors[f"{b1}-{b2}"] = self.detections.get_colors("quantile", (b1, b2), ml=True, flux=True, q=0.9)
        return pd.DataFrame(colors)
