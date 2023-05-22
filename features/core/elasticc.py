import numpy as np
import pandas as pd

from ._base import BaseFeatureExtractor
from .utils import decorators, extras


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
    * `spm`: Supernova parametric model parameters for each object, per band. Uses SPM v2 (multi-band fit).
    * `sn_features`: Magnitude and date statistics for detections and non-detections

    Notes:
        If `periods` is excluded, `COMPUTE_KIM`, `N_HARMONICS` and `POWER_RATE_FACTORS` will all be ignored, as they
        depend on values computed in the main period calculations.
    """

    NAME = "elasticc_lc_features"
    VERSION = "1.0"
    SURVEYS = ("LSST",)
    BANDS = ("u", "g", "r", "i", "z", "Y")  # Order matters for colors!
    EXTRA_COLUMNS = ["mwebv", "z_final"]
    CORRECTED = True
    FLUX = True
    MIN_DETECTIONS = 5
    MIN_DETECTIONS_IN_FID = 0

    def __init__(
        self,
        detections: list[dict] | pd.DataFrame,
        non_detections: list[dict] | pd.DataFrame = None,
        xmatches: list[dict] | pd.DataFrame = None,
        **kwargs,
    ):
        if kwargs.pop("legacy", False):
            metadata = kwargs.pop("metadata", None)
            detections, non_detections, xmatches = self._legacy(detections, non_detections, xmatches, metadata)

        super().__init__(detections, non_detections, xmatches)

        # Additional field required to compute SN features
        value, error = self.detections.alerts()[["mag_ml", "e_mag_ml"]].T.values
        # TODO: "3" is an arbitrary, should be replaced with a new one
        self.detections.add_field("detected", np.abs(value) - 3 * error > 0)

    @staticmethod
    def _legacy(detections, non_detections, xmatches, metadata):
        try:
            detections = detections.set_index("SNID")
        except KeyError:  # Assumes it is already indexed correctly
            pass
        if metadata is not None:
            try:
                metadata = metadata.set_index("SNID")
            except KeyError:  # Assumes it is already indexed correctly
                pass
            detections = detections.assign(mwebv=metadata["MWEBV"], z_final=metadata["REDSHIFT_HELIO"])
        detections = detections.reset_index()
        detections["sid"] = "LSST"
        detections["corrected"] = True
        detections = detections.rename(
            columns={
                "SNID": "aid",
                "FLUXCAL": "mag",
                "FLUXCALERR": "e_mag",
                "MJD": "mjd",
                "BAND": "fid",
            }
        )
        detections = detections.assign(mag_corr=detections["mag"], e_mag_corr_ext=detections["e_mag"])
        detections = detections.reset_index(names="candid")  # Fake candid

        if isinstance(non_detections, pd.DataFrame):
            raise NotImplemented("Legacy ELAsTiCC does not implement non-detections")
        if isinstance(xmatches, pd.DataFrame):
            raise NotImplemented("Legacy ELAsTiCC does not implement cross-match")

        return detections, non_detections, xmatches

    def _discard_detections(self):
        """Exclude noisy detections"""
        self.detections.select(["mag_ml", "e_mag_ml"], lt=[50e3, 300], gt=[-50e3, None])
        super()._discard_detections()

    @decorators.add_fid(0)
    def calculate_mwebv(self) -> pd.DataFrame:
        return pd.DataFrame({"mwebv": self.detections.agg("mwebv", "median")})

    @decorators.add_fid(0)
    def calculate_redshift_helio(self) -> pd.DataFrame:
        return pd.DataFrame({"redshift_helio": self.detections.agg("z_final", "median")})

    @decorators.add_fid("".join(BANDS))
    def calculate_colors(self) -> pd.DataFrame:
        colors = {}
        for b1, b2 in zip(self.BANDS[:-1], self.BANDS[1:]):
            colors[f"{b1}-{b2}"] = self.detections.get_colors("quantile", (b1, b2), ml=True, flux=self.FLUX, q=0.9)
        return pd.DataFrame(colors)

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid()
    def calculate_spm(self) -> pd.DataFrame:
        features = self.detections.apply(extras.fit_spm, version="v2", multiband=True, flux=self.FLUX, correct=True)
        return features.stack("fid")  # Needed for decorators to work

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid(counters="n_")
    def calculate_sn_features(self) -> pd.DataFrame:
        # Get mjd and flux of first detection of each object (any band)
        mjd = self.detections.agg("mjd", "min", flag="detected")
        flux = self.detections.which_value("mag_ml", which="first", flag="detected")
        return self.detections.apply(sn_features, first_mjd=mjd, first_flux=flux, by_fid=True)


def sn_features(df: pd.DataFrame, first_mjd: pd.Series, first_flux: pd.Series) -> pd.Series:
    aid, = df["id"].unique()  # Should never be called with more than one id at the time
    positive_fraction = (df["mag_ml"] > 0).mean()

    non_det_before = df[(df["mjd"] < first_mjd.loc[aid]) & ~df["detected"]]
    non_det_after = df[(df["mjd"] >= first_mjd.loc[aid]) & ~df["detected"]]

    n_non_det_before = non_det_before["mjd"].count()
    n_non_det_after = non_det_after["mjd"].count()

    try:
        last_flux_before = non_det_before["mag_ml"][non_det_before["mjd"].idxmax()]
    except ValueError:  # Occurs for empty non_det_before
        last_flux_before = np.nan
    max_flux_before = non_det_before["mag_ml"].max()
    median_flux_before = non_det_before["mag_ml"].median()

    dflux_first = first_flux.loc[aid] - last_flux_before
    dflux_median_before = first_flux.loc[aid] - median_flux_before

    max_flux_after = non_det_after["mag_ml"].max()
    median_flux_after = non_det_after["mag_ml"].median()

    return pd.Series(
        {
            "positive_fraction": positive_fraction,
            "dflux_first_det_band": dflux_first,
            "dflux_non_det_band": dflux_median_before,
            "last_flux_before_band": last_flux_before,
            "max_flux_before_band": max_flux_before,
            "max_flux_after_band": max_flux_after,
            "median_flux_before_band": median_flux_before,
            "median_flux_after_band": median_flux_after,
            "n_non_det_before_band": n_non_det_before,
            "n_non_det_after_band": n_non_det_after,
        }
    )
