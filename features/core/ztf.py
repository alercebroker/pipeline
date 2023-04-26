import pandas as pd
from scipy import stats
from astropy.coordinates import SkyCoord

from .utils import decorators, specials
from ._base import BaseFeatureExtractor


class ZTFClassifierFeatureExtractor(BaseFeatureExtractor):
    SURVEYS = ("ZTF",)
    BANDS = ("g", "r")
    BANDS_MAPPING = {"g": 1, "r": 2}
    EXTRAS = ["rb", "sgscore1"]
    MIN_DETECTIONS: int = 0
    MIN_DETECTIONS_IN_FID: int = 5
    MIN_REAL_BOGUS = 0.55
    MAX_SIGMA_MAGNITUDE = 1.0
    FATS_FEATURES = (
        "Amplitude",
        "AndersonDarling",
        "Autocor_length",
        "Beyond1Std",
        "Con",
        "Eta_e",
        "Gskew",
        "MaxSlope",
        "Mean",
        "Meanvariance",
        "MedianAbsDev",
        "MedianBRP",
        "PairSlopeTrend",
        "PercentAmplitude",
        "Q31",
        "Rcs",
        "Skew",
        "SmallKurtosis",
        "Std",
        "StetsonK",
        "Pvar",
        "ExcessVar",
        "SF_ML_amplitude",
        "SF_ML_gamma",
        "IAR_phi",
        "LinearTrend",
    )

    def _discard_detections(self):
        super()._discard_detections()
        self.detections.remove_alerts_out_of_range("rb", gt=self.MIN_REAL_BOGUS, ge=True)
        self.detections.remove_alerts_out_of_range("e_mag_ml", gt=0, lt=self.MAX_SIGMA_MAGNITUDE)

    def calculate_g_r_max(self) -> pd.DataFrame:
        """Uses `min` as aggregation function since it looks for max flux"""
        return pd.DataFrame({"g-r_max": self.detections.get_colors("min", ("g", "r"), ml=False)})

    def calculate_g_r_max_corr(self) -> pd.DataFrame:
        """Not necessarily corrected, still uses uncorrected for detections without correction"""
        return pd.DataFrame({"g-r_max_corr": self.detections.get_colors("min", ("g", "r"), ml=True)})

    def calculate_g_r_mean(self) -> pd.DataFrame:
        return pd.DataFrame({"g-r_mean": self.detections.get_colors("mean", ("g", "r"), ml=False)})

    def calculate_g_r_mean_corr(self) -> pd.DataFrame:
        return pd.DataFrame({"g-r_mean_corr": self.detections.get_colors("mean", ("g", "r"), ml=True)})

    def calculate_real_bogus(self) -> pd.DataFrame:
        return pd.DataFrame({"rb": self.detections.get_aggregate("rb", "median")})

    def calculate_sg_score(self) -> pd.DataFrame:
        return pd.DataFrame({"sgscore1": self.detections.get_aggregate("rb", "median")})

    def calculate_galactic_coordinates(self) -> pd.DataFrame:
        ra = self.detections.get_aggregate("ra", "mean")
        dec = self.detections.get_aggregate("dec", "mean")
        galactic = SkyCoord(ra, dec, frame="icrs", unit="deg").galactic
        # By construction, ra and dec indices should be the same
        return pd.DataFrame({"gal_b": galactic.b.degree, "gal_l": galactic.l.degree}, index=ra.index)

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid
    def calculate_fats(self) -> pd.DataFrame:
        return self.detections.apply_grouped(specials.fats4apply, by_fid=True, features=self.FATS_FEATURES)

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid
    def calculate_mhps(self) -> pd.DataFrame:
        return self.detections.apply_grouped(specials.mhps4apply, by_fid=True, t1=100, t2=10)

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid
    def calculate_iqr(self) -> pd.DataFrame:
        return pd.DataFrame({"iqr": self.detections.get_aggregate("mag_ml", stats.iqr, by_fid=True)})

    @decorators.columns_per_fid(BANDS_MAPPING)
    @decorators.fill_in_every_fid(fill_value=0)
    def calculate_n_pos(self) -> pd.DataFrame:
        return pd.DataFrame({"n_pos": self.detections.get_count_by_sign(1, bands=self.BANDS, by_fid=True)})

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid(fill_value=0)
    def calculate_n_neg(self) -> pd.DataFrame:
        return pd.DataFrame({"n_neg": self.detections.get_count_by_sign(-1, bands=self.BANDS, by_fid=True)})

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid(fill_value=0)
    def calculate_n_det(self) -> pd.DataFrame:
        n_pos = self.detections.get_count_by_sign(1, bands=self.BANDS, by_fid=True)
        n_neg = self.detections.get_count_by_sign(-1, bands=self.BANDS, by_fid=True)
        return pd.DataFrame({"n_det": n_pos + n_neg})

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid(fill_value=0)
    def calculate_positive_fraction(self) -> pd.DataFrame:
        n_pos = self.detections.get_count_by_sign(1, bands=self.BANDS, by_fid=True)
        n_neg = self.detections.get_count_by_sign(-1, bands=self.BANDS, by_fid=True)
        return pd.DataFrame({"positive_fraction": n_pos / (n_pos + n_neg)})

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid
    def calculate_delta_mjd(self) -> pd.DataFrame:
        return pd.DataFrame({"delta_mjd_fid": self.detections.get_delta("mjd", by_fid=True)})

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid
    def calculate_delta_mag(self) -> pd.DataFrame:
        return pd.DataFrame({"delta_mag_fid": self.detections.get_delta("mag_ml", by_fid=True)})

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid
    def calculate_min_mag(self) -> pd.DataFrame:
        return pd.DataFrame({"min_mag": self.detections.get_aggregate("mag_ml", "min", by_fid=True)})

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid
    def calculate_mean_mag(self) -> pd.DataFrame:
        return pd.DataFrame({"mean_mag": self.detections.get_aggregate("mag_ml", "mean", by_fid=True)})

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid
    def calculate_first_mag(self) -> pd.DataFrame:
        return pd.DataFrame({"first_mag": self.detections.get_which_value("mag_ml", which="first", by_fid=True)})

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid(fill_value=0)
    def calculate_n_non_det_before(self) -> pd.DataFrame:
        mjd = self.detections.get_aggregate("mjd", "min", by_fid=True)
        return pd.DataFrame(
            {
                "n_non_det_before_fid": self.non_detections.get_aggregate_when(
                    mjd, "mjd", "count", when="before", by_fid=True
                )
            }
        )

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid(fill_value=0)
    def calculate_n_non_det_after(self) -> pd.DataFrame:
        mjd = self.detections.get_aggregate("mjd", "min", by_fid=True)
        return pd.DataFrame(
            {
                "n_non_det_after_fid": self.non_detections.get_aggregate_when(
                    mjd, "mjd", "count", when="after", by_fid=True
                )
            }
        )

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid
    def calculate_max_diffmaglim_before(self) -> pd.DataFrame:
        mjd = self.detections.get_aggregate("mjd", "min", by_fid=True)
        return pd.DataFrame(
            {
                "max_diffmaglim_before_fid": self.non_detections.get_aggregate_when(
                    mjd, "diffmaglim", "max", when="before", by_fid=True
                )
            }
        )

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid
    def calculate_max_diffmaglim_after(self) -> pd.DataFrame:
        mjd = self.detections.get_aggregate("mjd", "min", by_fid=True)
        return pd.DataFrame(
            {
                "max_diffmaglim_after_fid": self.non_detections.get_aggregate_when(
                    mjd, "diffmaglim", "max", when="after", by_fid=True
                )
            }
        )

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid
    def calculate_median_diffmaglim_before(self) -> pd.DataFrame:
        mjd = self.detections.get_aggregate("mjd", "min", by_fid=True)
        return pd.DataFrame(
            {
                "median_diffmaglim_before_fid": self.non_detections.get_aggregate_when(
                    mjd, "diffmaglim", "median", when="before", by_fid=True
                )
            }
        )

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid
    def calculate_median_diffmaglim_after(self) -> pd.DataFrame:
        mjd = self.detections.get_aggregate("mjd", "min", by_fid=True)
        return pd.DataFrame(
            {
                "median_diffmaglim_after_fid": self.non_detections.get_aggregate_when(
                    mjd, "diffmaglim", "median", when="after", by_fid=True
                )
            }
        )

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid
    def calculate_last_mjd_before(self) -> pd.DataFrame:
        mjd = self.detections.get_aggregate("mjd", "min", by_fid=True)
        return pd.DataFrame(
            {
                "last_mjd_before_fid": self.non_detections.get_aggregate_when(
                    mjd, "mjd", "max", when="before", by_fid=True
                )
            }
        )

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid
    def calculate_last_diffmaglim_before(self) -> pd.DataFrame:
        mjd = self.detections.get_aggregate("mjd", "min", by_fid=True)
        return pd.DataFrame(
            {
                "last_diffmaglim_before_fid": self.non_detections.get_which_value_when(
                    mjd, "diffmaglim", which="last", when="before", by_fid=True
                )
            }
        )

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid
    def calculate_dmag_non_det(self) -> pd.DataFrame:
        mjd = self.detections.get_aggregate("mjd", "min", by_fid=True)
        diff = self.non_detections.get_aggregate_when(mjd, "diffmaglim", "median", when="before", by_fid=True)
        mag = self.detections.get_aggregate("mag_ml", "min", by_fid=True)
        return pd.DataFrame({"dmag_non_det_fid": diff - mag})

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid
    def calculate_dmag_first_det(self) -> pd.DataFrame:
        mjd = self.detections.get_aggregate("mjd", "min", by_fid=True)
        diff = self.non_detections.get_which_value_when(mjd, "diffmaglim", which="last", when="before", by_fid=True)
        mag = self.detections.get_which_value("mag_ml", which="first", by_fid=True)
        return pd.DataFrame({"dmag_first_det_fid": diff - mag})
