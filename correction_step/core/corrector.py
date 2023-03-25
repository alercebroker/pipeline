import numpy as np
import pandas as pd

from . import strategy


class Corrector:
    """Class for applying corrections"""
    _ZERO_MAG = 100.  # Not really zero mag, but zero flux (very high magnitude)

    def __init__(self, detections: list[dict], non_detections: list[dict]):
        self._non_detections = pd.DataFrame.from_records(non_detections)
        self._non_detections = self._non_detections.drop_duplicates(["oid", "fid", "mjd"])

        extras = [{**alert["extra_fields"], "candid": alert["candid"]} for alert in detections]
        extras_df = pd.DataFrame.from_records(extras, index="candid")
        self._extra_columns = extras_df.columns  # Only used to format output

        detections_df = pd.DataFrame.from_records(detections, exclude={"extra_fields"}, index="candid")
        # Remove duplicate detections by candid, always keeping those with stamps
        self._detections = detections_df.join(extras_df).sort_values("has_stamp", ascending=False)
        self._detections = self._detections[~self._detections.index.duplicated(keep="first")]

    def _survey_mask(self, survey: str):
        return self._detections["tid"].str.lower().str.startswith(survey.lower())

    def _apply_all_surveys(self, function: str, default=None, columns=None):
        if columns:
            basic = pd.DataFrame(default, index=self._detections.index, columns=columns)
        else:
            basic = pd.Series(default, index=self._detections.index)

        for name in strategy.__dict__:
            if name.startswith("_"):
                continue
            mask = self._survey_mask(name)
            if mask.any():
                module = getattr(strategy, name)
                basic[mask] = getattr(module, function)(self._detections[mask])

        return basic

    @property
    def corrected(self) -> pd.Series:
        """Whether the detection has a nearby known source"""
        return self._apply_all_surveys("is_corrected", False)

    @property
    def dubious(self) -> pd.Series:
        """Whether the correction or lack thereof is dubious"""
        return self._apply_all_surveys("is_dubious", False)

    @property
    def stellar(self) -> pd.Series:
        """Whether the source is likely stellar"""
        return self._apply_all_surveys("is_stellar", False)

    def _correct(self) -> pd.DataFrame:
        """Convert difference magnitude into apparent magnitude"""
        columns = ["mag_corr", "e_mag_corr", "e_mag_corr_ext"]
        return pd.DataFrame(columns=columns, index=self._detections.index)

    def corrected_dataframe(self) -> pd.DataFrame:
        """Alert dataframe including corrected magnitudes.

        Corrected magnitudes for objects considered far from a nearby source are set to NaN.
        """
        corrected = self._correct().replace(np.inf, self._ZERO_MAG)
        corrected[~self.corrected] = np.nan
        corrected = corrected.assign(corrected=self.corrected, dubious=self.dubious, stellar=self.stellar)

        return self._detections.join(corrected, lsuffix="_old").replace(np.nan, None)
