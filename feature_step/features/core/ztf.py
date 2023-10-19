import pandas as pd

from ._base import BaseFeatureExtractor
from .utils import decorators, functions, extras
import os
from importlib import metadata


class ZTFFeatureExtractor(BaseFeatureExtractor):
    """Extractor for ZTF light-curve classifier.

    Uses only alerts from ZTF in the bands g and r. For historical reasons, bands are mapped (g to 1 and r to 2).

    Uses corrected magnitudes when possible. Otherwise, defaults are the same as the base extractor.

    The extractors included (and computed) here by default are:

    * `galactic_coordinates`: Galactic latitude and longitude of each object.
    * `periods`: Period related features (multi-band period, per band period, etc.). Check notes.
    * `fats`: Computes all FATS features defined in `FATS_FEATURES`.
    * `mhps`: Mexican hat power spectrum for each object, per band.
    * `gp_drw`: Gaussian process damp random walk parameters for each object and band.
    * `iqr`: Inter-quartile range in magnitude distribution of each object, per band.
    * `colors`: Compute maximum and mean g-r colors for each object (both corrected, if available, and uncorrected)
    * `wise_colors`: W1-W2, W2-W3, g-W2, g-W3, r-W2, r-W3 mean colors for each object (if cross-match is available)
    * `real_bogus`: Median real-bogus value for each object
    * `sg_score`: Median start-galaxy score for each object
    * `spm`: Supernova parametric model parameters for each object, per band. Uses SPM v1 (per band fit).
    * `sn_features`: Magnitude and date statistics for detections and non-detections

    Notes:
        If `periods` is excluded, `COMPUTE_KIM`, `N_HARMONICS` and `POWER_RATE_FACTORS` will all be ignored, as they
        depend on values computed in the main period calculations.
    """

    NAME = "ztf_lc_features"
    VERSION = metadata.version("feature-step")
    SURVEYS = ("ZTF",)
    BANDS = ("g", "r")
    BANDS_MAPPING = {"g": 1, "r": 2}
    EXTRA_COLUMNS = ["ra", "dec", "isdiffpos", "rb", "sgscore1"]
    XMATCH_COLUMNS = {"allwise": ["W1mag", "W2mag", "W3mag"]}
    CORRECTED = True
    MIN_REAL_BOGUS = 0.55
    MAX_ERROR = 1.0

    def __init__(
        self,
        detections: list[dict] | pd.DataFrame,
        non_detections: list[dict] | pd.DataFrame = None,
        xmatches: list[dict] | pd.DataFrame = None,
        **kwargs,
    ):
        if kwargs.pop("legacy", False):
            metadata = kwargs.pop("metadata", None)
            detections, non_detections, xmatches = self._legacy(
                detections, non_detections, xmatches, metadata
            )

        detections = list(
            filter(
                lambda d: not d.get("forced", False) and d["sid"] == "ZTF",
                detections,
            )
        )
        super().__init__(detections, non_detections, xmatches)

        # Change isdiffpos from 1 or -1 to True or False
        self.detections.add_field(
            "isdiffpos", self.detections.alerts()["isdiffpos"] > 0
        )

    @classmethod
    def _legacy(cls, detections, non_detections, xmatches, metadata):
        try:
            detections = detections.set_index("oid")
        except KeyError:  # Assumes it is already indexed correctly
            pass
        if metadata is not None:
            try:
                metadata = metadata.set_index("oid")
            except KeyError:  # Assumes it is already indexed correctly
                pass
            detections = detections.assign(sgscore1=metadata["sgscore1"])
        detections = detections.reset_index()
        detections["fid"].replace(
            {v: k for k, v in cls.BANDS_MAPPING.items()}, inplace=True
        )  # reverse mapping

        if isinstance(non_detections, pd.DataFrame):
            non_detections = non_detections.reset_index()
            non_detections["sid"] = "ZTF"
            non_detections["fid"].replace(
                {v: k for k, v in cls.BANDS_MAPPING.items()}, inplace=True
            )
            non_detections = non_detections.rename(columns={"oid": "aid"})

        if isinstance(xmatches, pd.DataFrame):
            xmatches = xmatches.reset_index()
            xmatches = xmatches.rename(columns={"oid": "aid"})

        return detections, non_detections, xmatches

    def _discard_detections(self):
        """Include only alerts with a minimum real-bogus value and a maximum error in magnitude"""
        self.logger.debug(
            f"Selecting detections with real/bogus score greater than {self.MIN_REAL_BOGUS}"
        )
        self.detections.select("rb", gt=self.MIN_REAL_BOGUS)
        self.logger.debug(
            f"{len(self.detections.alerts())} detections remain after selection"
        )

        self.logger.debug(
            f"Selecting detections with magnitude error lower than {self.MAX_ERROR}"
        )
        self.detections.select("e_mag_ml", gt=0, lt=self.MAX_ERROR)
        self.logger.debug(
            f"{len(self.detections.alerts())} detections remain after selection"
        )
        super()._discard_detections()

    @decorators.logger
    @decorators.add_fid("")
    def calculate_galactic_coordinates(self) -> pd.DataFrame:
        ra = self.detections.agg("ra", "mean")
        dec = self.detections.agg("dec", "mean")
        return extras.galactic_coordinates(ra, dec, frame="icrs")

    @decorators.logger
    @decorators.add_fid("gr")
    def calculate_colors(self) -> pd.DataFrame:
        gr_max = self.detections.get_colors(
            "min", ("g", "r"), ml=False, flux=self.FLUX
        )
        gr_max_corr = self.detections.get_colors(
            "min", ("g", "r"), ml=True, flux=self.FLUX
        )
        gr_mean = self.detections.get_colors(
            "mean", ("g", "r"), ml=False, flux=self.FLUX
        )
        gr_mean_corr = self.detections.get_colors(
            "mean", ("g", "r"), ml=True, flux=self.FLUX
        )
        return pd.DataFrame(
            {
                "g-r_max": gr_max,
                "g-r_max_corr": gr_max_corr,
                "g-r_mean": gr_mean,
                "g-r_mean_corr": gr_mean_corr,
            }
        )

    @decorators.logger
    @decorators.add_fid("")
    def calculate_wise_colors(self) -> pd.DataFrame:
        if self.FLUX:
            raise ValueError("Cannot calculate WISE colors with flux.")
        mags = functions.fill_index(
            self.detections.agg("mag_ml", "mean", by_fid=True), fid=("g", "r")
        )
        g, r = mags.xs("g", level="fid"), mags.xs("r", level="fid")
        return pd.DataFrame(
            {
                "W1-W2": self.xmatches["W1mag"] - self.xmatches["W2mag"],
                "W2-W3": self.xmatches["W2mag"] - self.xmatches["W3mag"],
                "g-W2": g - self.xmatches["W2mag"],
                "g-W3": g - self.xmatches["W3mag"],
                "r-W2": r - self.xmatches["W2mag"],
                "r-W3": r - self.xmatches["W3mag"],
            }
        )

    @decorators.logger
    @decorators.add_fid("")
    def calculate_real_bogus(self) -> pd.DataFrame:
        return pd.DataFrame({"rb": self.detections.agg("rb", "median")})

    @decorators.logger
    @decorators.add_fid("")
    def calculate_sg_score(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"sgscore1": self.detections.agg("sgscore1", "median")}
        )

    @decorators.logger
    @decorators.columns_per_fid
    @decorators.fill_in_every_fid()
    def calculate_spm(self) -> pd.DataFrame:
        # TODO: Bug option is included because model is trained with the bug. Should be removed if retrained
        return self.detections.apply(
            extras.fit_spm,
            by_fid=True,
            version="v1",
            bug=True,
            ml=False,
            flux=self.FLUX,
        )

    @decorators.logger
    @decorators.columns_per_fid
    @decorators.fill_in_every_fid(counters="n_")
    def calculate_sn_features(self):
        n_pos = self.detections.agg("isdiffpos", "sum", by_fid=True)
        n_det = self.detections.agg("isdiffpos", "count", by_fid=True)
        n_neg = n_det - n_pos
        positive_fraction = n_pos / n_det

        n_non_det_before = self.non_detections.agg_when(
            "mjd", "count", when="before", by_fid=True
        )
        n_non_det_after = self.non_detections.agg_when(
            "mjd", "count", when="after", by_fid=True
        )

        delta_mjd = self.detections.get_delta("mjd", by_fid=True)
        delta_mag = self.detections.get_delta("mag_ml", by_fid=True)
        min_mag = self.detections.agg("mag_ml", "min", by_fid=True)
        mean_mag = self.detections.agg("mag_ml", "mean", by_fid=True)
        first_mag = self.detections.which_value(
            "mag_ml", which="first", by_fid=True
        )

        max_upper_before = self.non_detections.agg_when(
            "diffmaglim", "max", when="before", by_fid=True
        )
        max_upper_after = self.non_detections.agg_when(
            "diffmaglim", "max", when="after", by_fid=True
        )
        median_upper_before = self.non_detections.agg_when(
            "diffmaglim", "median", when="before", by_fid=True
        )
        median_upper_after = self.non_detections.agg_when(
            "diffmaglim", "median", when="after", by_fid=True
        )
        last_mjd_before = self.non_detections.agg_when(
            "mjd", "max", when="after", by_fid=True
        )
        last_upper_before = self.non_detections.which_value_when(
            "diffmaglim", which="last", when="after", by_fid=True
        )
        dmag_non_det = median_upper_before - min_mag
        dmag_first_det = last_upper_before - first_mag

        return pd.DataFrame(
            {
                "n_det": n_det,
                "n_neg": n_neg,
                "n_pos": n_pos,
                "n_non_det_before_fid": n_non_det_before,
                "n_non_det_after_fid": n_non_det_after,
                "delta_mag_fid": delta_mag,
                "delta_mjd_fid": delta_mjd,
                "first_mag": first_mag,
                "mean_mag": mean_mag,
                "min_mag": min_mag,
                "positive_fraction": positive_fraction,
                "max_diffmaglim_before_fid": max_upper_before,
                "median_diffmaglim_before_fid": median_upper_before,
                "last_diffmaglim_before_fid": last_upper_before,
                "last_mjd_before_fid": last_mjd_before,
                "dmag_non_det_fid": dmag_non_det,
                "dmag_first_det_fid": dmag_first_det,
                "max_diffmaglim_after_fid": max_upper_after,
                "median_diffmaglim_after_fid": median_upper_after,
            }
        )
