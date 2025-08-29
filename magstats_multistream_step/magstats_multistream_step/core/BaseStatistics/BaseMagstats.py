
from typing import List
import pandas as pd

from ._base import BaseStatistics

class BaseMagnitudeStatistics(BaseStatistics):
    """Base class for magnitude-related statistics - corresponds to the Base MagStats table only
       Specific survey statistics are handled by the survey specifc statistics"""
    _JOIN = ["oid", "band"]  

    def __init__(self, detections: List[dict], non_detections: List[dict] = None):
        super().__init__(detections)
        if non_detections:
            self._non_detections = pd.DataFrame.from_records(
                non_detections
            ).drop_duplicates(["oid", "band", "mjd"])
        else:
            self._non_detections = pd.DataFrame()
    
    # Base magstats functions calculate the data for ndet, firstmjd and lastmjd only!

    def calculate_ndet(self) -> pd.DataFrame:
        return pd.DataFrame({
            "ndet": self._detections.value_counts(subset=self._JOIN, sort=False, dropna=False)
        })
    
    def calculate_firstmjd(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"firstmjd": self._grouped_value("mjd", which="first")}
        )
    
    def calculate_lastmjd(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"lastmjd": self._grouped_value("mjd", which="last")}
        )
    