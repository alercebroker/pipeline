import warnings
from typing import List

import pandas as pd

from ._base import BaseStatistics


class MagnitudeStatistics(BaseStatistics):
    _JOIN = ["aid", "fid"]
    MAGNITUDE_THRESHOLD = 13.2

    def __init__(self, detections: List[dict], non_detections: List[dict]):
        super().__init__(detections)
        if non_detections:
            self._non_detections = pd.DataFrame.from_records(non_detections).drop_duplicates(["oid", "fid", "mjd"])
        else:
            self._non_detections = pd.DataFrame()

    def _calculate_stats(self, corrected: bool = False) -> pd.DataFrame:
        suffix = "_corr" if corrected else ""

        grouped = self._grouped_detections(corrected=corrected)
        functions = {"mean": "mean", "median": "median", "max": "max", "min": "min", "sigma": "std"}
        functions = {f"mag{k}{suffix}": v for k, v in functions.items()}

        aggregated = grouped[f"mag{suffix}"].agg(**functions)
        return aggregated["sigma"].fillna(0)  # pandas std gives NaN if there is only one sample

    def _calculate_stats_over_time(self, corrected: bool = False) -> pd.DataFrame:
        suffix = "_corr" if corrected else ""
        first_mag = self._grouped_value(f"mag{suffix}", which="first", corrected=corrected)
        last_mag = self._grouped_value(f"mag{suffix}", which="last", corrected=corrected)
        return pd.DataFrame({f"magfirst{suffix}": first_mag, f"maglast{suffix}": last_mag})

    def calculate_stats(self) -> pd.DataFrame:
        stats = self._calculate_stats(corrected=False)
        stats = stats.join(self._calculate_stats_over_time(corrected=False))
        stats = stats.join(self._calculate_stats(corrected=True))
        return stats.join(self._calculate_stats_over_time(corrected=True))

    def calculate_firstmjd(self) -> pd.DataFrame:
        return pd.DataFrame({"firstmjd": self._grouped_value("mjd", which="first")})

    def calculate_lastmjd(self) -> pd.DataFrame:
        return pd.DataFrame({"lastmjd": self._grouped_value("mjd", which="last")})

    def calculate_corrected(self) -> pd.DataFrame:
        return pd.DataFrame({"corrected": self._grouped_value("corrected", which="first")})

    def calculate_stellar(self) -> pd.DataFrame:
        return pd.DataFrame({"stellar": self._grouped_value("stellar", which="first")})

    def calculate_ndet(self) -> pd.DataFrame:
        # The column selected for ndet is irrelevant as long as it has no NaN values
        return pd.DataFrame({"ndet": self._grouped_detections()["oid"].count()})

    def calculate_ndubious(self) -> pd.DataFrame:
        return pd.DataFrame({"ndubious": self._grouped_detections()["dubious"].sum()})

    def calculate_saturation_rate(self) -> pd.DataFrame:
        mask = self._detections["mag_corr"] < self.MAGNITUDE_THRESHOLD
        saturated = self._group(self._detections[mask])["mag_corr"].count()
        total = self._grouped_detections()["mag_corr"].count()  # Count also excludes NaNs
        with warnings.catch_warnings():
            # possible 0 divided by 0; this is expected and returned NaN is correct value
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return pd.DataFrame({"saturation_rate": saturated / total})

    def calculate_dmdt(self) -> pd.DataFrame:
        dt_min = 0.5

        if self._non_detections.size == 0:  # Handle no non-detection case
            return pd.DataFrame(columns=["dt_first", "dm_first", "sigmadm_first", "dmdt_first"])

        first_mag = self._grouped_value("mag", which="first")
        first_e_mag = self._grouped_value("e_mag", which="first")
        first_mjd = self._grouped_value("mjd", which="first")

        nd = self._non_detections.set_index(self._JOIN)  # Index by join to compute based on it

        dt = first_mjd - nd["mjd"]
        dm = first_mag - nd["diffmaglim"]
        sigmadm = first_e_mag - nd["diffmaglim"]
        dmdt = (first_mag + first_e_mag - nd["diffmaglim"]) / dt

        # Include back fid for grouping and unique identification
        results = pd.DataFrame({"dt": dt, "dm": dm, "sigmadm": sigmadm, "dmdt": dmdt}).reset_index()
        # Only include non-detections before dt_min
        idx = self._group(results[results["dt"] > dt_min])["dmdt"].idxmin().dropna()

        # Drop NaN, since they result from no non-detection before first detection
        results = results.dropna().loc[idx].set_index("fid")
        return results.rename(columns={c: f"{c}_first" for c in results.columns})
