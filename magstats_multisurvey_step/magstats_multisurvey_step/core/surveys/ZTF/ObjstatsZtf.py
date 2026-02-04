from typing import Union, Literal

import pandas as pd
from ...BaseStatistics.BaseObjectstats import BaseObjectStatistics
from ...StatisticsSelector.statistics_selector import register_survey_class_objstat

# Survey-specific implementations
class ZTFObjectStatistics(BaseObjectStatistics):
    """ZTF-specific object statistics"""
    _CORRECTED = ("ZTF",)
    _STELLAR = ("ZTF",)
    
    def calculate_corrected(self) -> pd.DataFrame:
        return pd.DataFrame({
            "corrected": self._grouped_detections(surveys=self._CORRECTED)["corrected"].any()
    })

    def calculate_stellar(self) -> pd.DataFrame:
        return pd.DataFrame({
            "stellar": self._grouped_detections(surveys=self._STELLAR)["stellar"].any()
        })
    
    def calculate_reference_change(self) -> pd.DataFrame:
        return pd.DataFrame({
            "reference_change": self._grouped_value("mjdendref", which="last")
                               > self._grouped_value("mjd", which="first")
        })
    
    def calculate_diffpos(self) -> pd.DataFrame:
        return pd.DataFrame({
            "diffpos": self._grouped_value("isdiffpos", which="first") > 0
        })

register_survey_class_objstat("ZTF", ZTFObjectStatistics)
