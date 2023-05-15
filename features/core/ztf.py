import pandas as pd

from ._base import BaseFeatureExtractor
from .utils import decorators, functions, extras
from .utils.extras.spm import fit_spm_v1


class ZTFClassifierFeatureExtractor(BaseFeatureExtractor):
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
    * `spm`: Supernova parametric model parameters for each object, per band. Uses SPM v1.
    * `sn_stats`: Magnitude and date statistics for detections and non-detections
    * `counters`: Number of detections (positive and negative) and non-detections (before and after first detection)

    Notes:
        If `periods` is excluded, `COMPUTE_KIM`, `N_HARMONICS` and `POWER_RATE_FACTORS` will all be ignored, as they
        depend on values computed in the main period calculations.
    """

    NAME = "ztf_lc_features"
    VERSION = "1.0"
    SURVEYS = ("ZTF",)
    BANDS = ("g", "r")
    BANDS_MAPPING = {"g": 1, "r": 2, "gr": 12}
    EXTRA_COLUMNS = ["rb", "sgscore1"]
    XMATCH_COLUMNS = ["W1mag", "W2mag", "W3mag"]
    USE_CORRECTED = True
    MIN_REAL_BOGUS = 0.55
    MAX_SIGMA_MAGNITUDE = 1.0

    def _discard_detections(self):
        """In addition to base checks, keeps only alerts with a minimum real-bogus value and a maximum
        error in magnitude"""
        self.detections.select("rb", gt=self.MIN_REAL_BOGUS, ge=True)
        self.detections.select("e_mag_ml", gt=0, lt=self.MAX_SIGMA_MAGNITUDE)
        super()._discard_detections()

    @decorators.add_fid(12)
    def calculate_colors(self) -> pd.DataFrame:
        gr_max = self.detections.get_colors("min", ("g", "r"), ml=False)
        gr_max_corr = self.detections.get_colors("min", ("g", "r"), ml=True)
        gr_mean = self.detections.get_colors("mean", ("g", "r"), ml=False)
        gr_mean_corr = self.detections.get_colors("mean", ("g", "r"), ml=True)
        return pd.DataFrame(
            {"g-r_max": gr_max, "g-r_max_corr": gr_max_corr, "g-r_mean": gr_mean, "g-r_mean_corr": gr_mean_corr}
        )

    @decorators.add_fid(0)
    def calculate_wise_colors(self) -> pd.DataFrame:
        mags = functions.fill_index(self.detections.get_aggregate("mag_ml", "mean", by_fid=True), fid=("g", "r"))
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

    @decorators.add_fid(0)
    def calculate_real_bogus(self) -> pd.DataFrame:
        return pd.DataFrame({"rb": self.detections.get_aggregate("rb", "median")})

    @decorators.add_fid(0)
    def calculate_sg_score(self) -> pd.DataFrame:
        return pd.DataFrame({"sgscore1": self.detections.get_aggregate("sgscore1", "median")})

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid()
    def calculate_spm(self) -> pd.DataFrame:
        return self.detections.apply_grouped(extras.spm_ztf, by_fid=True, spm=fit_spm_v1)

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid()
    def calculate_sn_stats(self):
        n_pos = self.detections.get_count_by_sign(1, bands=self.BANDS, by_fid=True)
        n_neg = self.detections.get_count_by_sign(-1, bands=self.BANDS, by_fid=True)
        positive_fraction = n_pos / n_pos + n_neg

        delta_mjd = self.detections.get_delta("mjd", by_fid=True)
        delta_mag = self.detections.get_delta("mag_ml", by_fid=True)
        min_mag = self.detections.get_aggregate("mag_ml", "min", by_fid=True)
        mean_mag = self.detections.get_aggregate("mag_ml", "mean", by_fid=True)
        first_mag = self.detections.get_which_value("mag_ml", which="first", by_fid=True)

        max_diff_bef = self.non_detections.get_aggregate_when("diffmaglim", "max", when="before", by_fid=True)
        max_diff_af = self.non_detections.get_aggregate_when("diffmaglim", "max", when="after", by_fid=True)
        med_diff_bef = self.non_detections.get_aggregate_when("diffmaglim", "median", when="before", by_fid=True)
        med_diff_af = self.non_detections.get_aggregate_when("diffmaglim", "median", when="after", by_fid=True)
        last_mjd_bef = self.non_detections.get_aggregate_when("mjd", "max", when="after", by_fid=True)
        last_diff_bef = self.non_detections.get_which_value_when("diffmaglim", which="last", when="after", by_fid=True)
        dmag_non_det = med_diff_bef - min_mag
        dmag_first_det = last_diff_bef - first_mag

        return pd.DataFrame(
            {
                "delta_mag_fid": delta_mag,
                "delta_mjd_fid": delta_mjd,
                "first_mag": first_mag,
                "mean_mag": mean_mag,
                "min_mag": min_mag,
                "positive_fraction": positive_fraction,
                "max_diffmaglim_before_fid": max_diff_bef,
                "median_diffmaglim_before_fid": med_diff_bef,
                "last_diffmaglim_before_fid": last_diff_bef,
                "last_mjd_before_fid": last_mjd_bef,
                "dmag_non_det_fid": dmag_non_det,
                "dmag_first_det_fid": dmag_first_det,
                "max_diffmaglim_after_fid": max_diff_af,
                "median_diffmaglim_after_fid": med_diff_af,
            }
        )

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid(fill_value=0, dtype=int)
    def calculate_counters(self):
        n_pos = self.detections.get_count_by_sign(1, bands=self.BANDS, by_fid=True)
        n_neg = self.detections.get_count_by_sign(-1, bands=self.BANDS, by_fid=True)
        n_det = n_pos + n_neg

        n_ndet_bef = self.non_detections.get_aggregate_when("mjd", "count", when="before", by_fid=True)
        n_ndet_af = self.non_detections.get_aggregate_when("mjd", "count", when="after", by_fid=True)

        return pd.DataFrame(
            {
                "n_det": n_det,
                "n_neg": n_neg,
                "n_pos": n_pos,
                "n_non_det_before_fid": n_ndet_bef,
                "n_non_det_after_fid": n_ndet_af,
            }
        ).fillna(0)  # Number of non-detections get filled by nan if there are none
