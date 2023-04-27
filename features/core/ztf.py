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
    USE_CORRECTED = True
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

    def calculate_colors(self) -> pd.DataFrame:
        gr_max = self.detections.get_colors("min", ("g", "r"), ml=False)
        gr_max_corr = self.detections.get_colors("min", ("g", "r"), ml=True)
        gr_mean = self.detections.get_colors("mean", ("g", "r"), ml=False)
        gr_mean_corr = self.detections.get_colors("mean", ("g", "r"), ml=True)
        return pd.DataFrame({"g-r_max": gr_max, "g-r_max_corr": gr_max_corr, "g-r_mean": gr_mean, "g-r_mean_corr": gr_mean_corr})

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
    @decorators.fill_in_every_fid()
    def calculate_fats(self) -> pd.DataFrame:
        return self.detections.apply_grouped(specials.fats4apply, by_fid=True, features=self.FATS_FEATURES)

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid()
    def calculate_mhps(self) -> pd.DataFrame:
        return self.detections.apply_grouped(specials.mhps4apply, by_fid=True, t1=100, t2=10)

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid()
    def calculate_iqr(self) -> pd.DataFrame:
        return pd.DataFrame({"iqr": self.detections.get_aggregate("mag_ml", stats.iqr, by_fid=True)})

    @decorators.columns_per_fid
    @decorators.fill_in_every_fid()
    def calculate_detections_and_non_detections(self):
        n_pos = self.detections.get_count_by_sign(1, bands=self.BANDS, by_fid=True)
        n_neg = self.detections.get_count_by_sign(1, bands=self.BANDS, by_fid=True)
        n_det = n_pos + n_neg
        positive_fraction = n_pos / n_det

        delta_mjd = self.detections.get_delta("mjd", by_fid=True)
        delta_mag = self.detections.get_delta("mag_ml", by_fid=True)
        min_mag = self.detections.get_aggregate("mag_ml", "min", by_fid=True)
        mean_mag = self.detections.get_aggregate("mag_ml", "mean", by_fid=True)
        first_mag = self.detections.get_which_value("mag_ml", which="first", by_fid=True)

        n_ndet_bef = self.non_detections.get_aggregate_when("mjd", "count", when="before", by_fid=True)
        n_ndet_af = self.non_detections.get_aggregate_when("mjd", "count", when="after", by_fid=True)
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
                "n_det": n_det,
                "n_neg": n_neg,
                "n_pos": n_pos,
                "positive_fraction": positive_fraction,
                "n_non_det_before_fid": n_ndet_bef,
                "max_diffmaglim_before_fid": max_diff_bef,
                "median_diffmaglim_before_fid": med_diff_bef,
                "last_diffmaglim_before_fid": last_diff_bef,
                "last_mjd_before_fid": last_mjd_bef,
                "dmag_non_det_fid": dmag_non_det,
                "dmag_first_det_fid": dmag_first_det,
                "n_non_det_after_fid": n_ndet_af,
                "max_diffmaglim_after_fid": max_diff_af,
                "median_diffmaglim_after_fid": med_diff_af,
            }
        )
