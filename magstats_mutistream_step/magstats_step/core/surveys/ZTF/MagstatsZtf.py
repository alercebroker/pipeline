from typing import List

import numpy as np
import pandas as pd

from ...BaseStatistics.BaseMagstats import BaseMagnitudeStatistics
from ...StatisticsSelector.statistics_selector import register_survey_class_magstat


class ZTFMagnitudeStatistics(BaseMagnitudeStatistics):
    """ZTF-specific magnitude statistics - corresponds to ztf_MagStats table"""
    _THRESHOLD = {"ZTF": 13.2}  # Saturation threshold for ZTF
    
    def _calculate_stats(self, corrected: bool = False) -> pd.DataFrame:
        """Calculate magnitude statistics (mean, median, max, min, sigma)"""
        suffix = "_corr" if corrected else ""
        in_label, out_label = f"mag{suffix}", f"mag{{}}{suffix}"
        
        grouped = self._grouped_detections(corrected=corrected)
        functions = {"mean", "median", "max", "min"}
        functions = {out_label.format(func): func for func in functions}
        
        stats = grouped[in_label].agg(**functions)
        return stats.join(
            grouped[in_label]
            .agg("std", ddof=0)
            .rename(out_label.format("sigma")),
            how="outer",
        )
    
    def _calculate_stats_over_time(self, corrected: bool = False) -> pd.DataFrame:
        """Calculate first and last magnitude"""
        suffix = "_corr" if corrected else ""
        in_label, out_label = f"mag{suffix}", f"mag{{}}{suffix}"
        
        first = self._grouped_value(in_label, which="first", corrected=corrected)
        last = self._grouped_value(in_label, which="last", corrected=corrected)
        return pd.DataFrame({
            out_label.format("first"): first, 
            out_label.format("last"): last
        })
    
    # ZTF-specific statistics corresponding to ztf_MagStats table
    def calculate_corrected(self) -> pd.DataFrame:
        return pd.DataFrame({
            "corrected": self._grouped_value("corrected", which="first")
        })
    
    def calculate_stellar(self) -> pd.DataFrame:
        return pd.DataFrame({
            "stellar": self._grouped_value("stellar", which="first")
        })
    
    def calculate_ndubious(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ndubious": self._grouped_detections()["dubious"].sum()
        })
    
    def calculate_magstats(self) -> pd.DataFrame:
        """Calculate all magnitude statistics (both corrected and uncorrected)"""
        stats = self._calculate_stats(corrected=False)
        stats = stats.join(
            self._calculate_stats_over_time(corrected=False), how="outer"
        )
        stats = stats.join(self._calculate_stats(corrected=True), how="outer")
        return stats.join(
            self._calculate_stats_over_time(corrected=True), how="outer"
        )
    
    def calculate_saturation_rate(self) -> pd.DataFrame:
        total = self._grouped_detections()["corrected"].sum()
        saturated = pd.Series(index=total.index, dtype=float)
        
        for survey, threshold in self._THRESHOLD.items():
            sat = self._grouped_detections(surveys=(survey,))["mag_corr"].agg(
                lambda x: (x < threshold).sum()
            )
            saturated.loc[sat.index] = sat
        
        rate = np.where(total.ne(0), saturated.astype(float) / total, np.nan)
        return pd.DataFrame({"saturation_rate": rate}, index=total.index)


    def calculate_dmdt(self) -> pd.DataFrame:
        dt_min = 0.5
        
        if self._non_detections.size == 0:
            return pd.DataFrame(
                columns=["dt_first", "dm_first", "sigmadm_first", "dmdt_first"]
            )
        
        first_mag = self._grouped_value("mag", which="first")
        first_e_mag = self._grouped_value("e_mag", which="first")
        first_mjd = self._grouped_value("mjd", which="first")
        
        nd = self._non_detections.set_index(self._JOIN)
        
        dt = first_mjd - nd["mjd"]
        dm = first_mag - nd["diffmaglim"]
        sigmadm = first_e_mag - nd["diffmaglim"]
        dmdt = (first_mag + first_e_mag - nd["diffmaglim"]) / dt
        
        results = pd.DataFrame({
            "dt": dt, "dm": dm, "sigmadm": sigmadm, "dmdt": dmdt
        }).reset_index()
        
        idx = (
            self._group(results[results["dt"] > dt_min])["dmdt"]
            .idxmin()
            .dropna()
        )
        
        results = results.dropna().loc[idx].set_index(self._JOIN)
        return results.rename(columns={c: f"{c}_first" for c in results.columns})
    

register_survey_class_magstat("ZTF", ZTFMagnitudeStatistics)
