import pandas as pd
import numpy as np
from typing import Union, Literal, List

from ._base import BaseStatistics

_TID_MAPPING = {
    "lsst": 0,
    "ztf": 1
}

class BaseObjectStatistics(BaseStatistics):
    """Base class for object statistics common to all surveys"""
    _JOIN = "oid"

    def __init__(self, detections: List[dict], non_detections: List[dict] = None, forced_detections: List[dict] = None, survey: str = None):
        super().__init__(detections)
        self._detections['tid'] = _TID_MAPPING.get(survey)
        # Create list with non detections to count for the magstats
        if non_detections:
            self._non_detections = pd.DataFrame.from_records(
                non_detections
            ).drop_duplicates(["oid", "band", "mjd"])
        else:
            self._non_detections = pd.DataFrame()
        # Create list with forced detections to count for the magstats 
        if forced_detections:
            self._forced_detections = pd.DataFrame.from_records(
                forced_detections
            ).drop_duplicates(["oid", "measurement_id"])
        else:
            self._forced_detections = pd.DataFrame()
        self.survey = survey
    
    @staticmethod
    def _arcsec2deg(values: Union[pd.Series, float]) -> Union[pd.Series, float]:
        return values / 3600.0
    
    @staticmethod
    def _deg2arcsec(values: Union[pd.Series, float]) -> Union[pd.Series, float]:
        return values * 3600.0
    
    @staticmethod
    def _compute_weights(sigmas: Union[pd.Series, float]) -> Union[pd.Series, float]:
        return sigmas.astype(float) ** -2
    
    @classmethod
    def _weighted_mean(cls, values: pd.Series, sigmas: pd.Series) -> float:
        return np.average(values, weights=cls._compute_weights(sigmas))
    
    @classmethod
    def _weighted_mean_error(cls, sigmas: pd.Series) -> float:
        return np.sqrt(1 / np.sum(cls._compute_weights(sigmas)))
    
    def _calculate_coordinates(self, label: Literal["ra", "dec"]) -> pd.DataFrame:
        def average(series):
            return self._weighted_mean(series, sigmas.loc[series.index])
        
        sigmas = self._arcsec2deg(self._detections[f"{label}_error"])
        grouped_sigmas = self._group(sigmas.set_axis(self._detections["oid"]))
        return pd.DataFrame(
            {
                f"mean{label}": self._grouped_detections()[label].agg(average),
                f"sigma{label}": self._deg2arcsec(
                    grouped_sigmas.agg(self._weighted_mean_error)
                ),
            }
        )
    
    def _calculate_unique(self, label: str) -> pd.DataFrame:
        grouped = self._grouped_detections()
        return grouped[label].agg(lambda x: x.iloc[0]).to_frame(name=label) 
    
    
    def calculate_ra(self) -> pd.DataFrame:
        return self._calculate_coordinates("ra")
    
    def calculate_dec(self) -> pd.DataFrame:
        return self._calculate_coordinates("dec")
    
    def calculate_firstmjd(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"firstmjd": self._grouped_value("mjd", which="first")}
        )
    
    def calculate_lastmjd(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"lastmjd": self._grouped_value("mjd", which="last")}
        )
    
    def calculate_deltajd(self) -> pd.DataFrame:
        return pd.DataFrame({
            "deltajd": self._grouped_value("mjd", which="last")
                      - self._grouped_value("mjd", which="first")
        })
    
    def calculate_oid(self) -> pd.DataFrame:
        return self._calculate_unique("oid")


    def calculate_tid(self) -> pd.DataFrame:
        return self._calculate_unique("tid")
    

    def calculate_sid(self) -> pd.DataFrame:
        return self._calculate_unique("sid")

    def calculate_ndet(self) -> pd.DataFrame:
        return pd.DataFrame({
            "n_det": self._detections.value_counts(subset=self._JOIN, sort=False)
        })

    
    def calculate_nndet(self) -> pd.DataFrame:
        # Get all oids  from detections to ensure we have all OIDs since some oids might not have non detections
        all_oids = self._detections.groupby(self._JOIN).size().index
        
        # Try to count non detections only if they exist .i.e there's at least a non detection that is not empty
        if len(self._non_detections) > 0:
            ndet_counts = self._non_detections.value_counts(subset=self._JOIN, sort=False)
        else:
            # If no nondetections, then create empty series and use the same index as detections which has all oids
            ndet_counts = pd.Series([], dtype='int64', name='count')
            ndet_counts.index = all_oids[:0] 
        
        # Filling missing oids with 0
        ndet_counts = ndet_counts.reindex(all_oids, fill_value=0)
        
        return pd.DataFrame({
            "n_ndet": ndet_counts
        })

    
    def calculate_nfphot(self) -> pd.DataFrame:
        # Get all oids  from detections to ensure we have all OIDs since some oids might not have forced phots
        all_oids = self._detections.groupby(self._JOIN).size().index
        
        # Try to count forced phots only if they exist .i.e there's at least a forced phots that is not empty
        if len(self._forced_detections) > 0:
            fphot_counts = self._forced_detections.value_counts(subset=self._JOIN, sort=False)
        else:
            # If no forced phots, then create empty series and use the same index as detections which has all oids
            fphot_counts = pd.Series([], dtype='int64', name='count')
            fphot_counts.index = all_oids[:0] 
        
        # Filling missing oids with 0
        fphot_counts = fphot_counts.reindex(all_oids, fill_value=0)
        
        return pd.DataFrame({
            "n_fphot": fphot_counts
        })
