from typing import Union, Literal

import pandas as pd
from ...BaseStatistics.BaseObjectstats import BaseObjectStatistics
from ...StatisticsSelector.statistics_selector import register_survey_class_objstat

class LSSTObjectStatistics(BaseObjectStatistics):
    """LSST object statistics - only base stats"""

    # Since we apply no corrections to LSST detections, all objects are not corrected for now
    def calculate_corrected(self) -> pd.DataFrame:
        # Get all unique oids from detections so each oid in objstats will have a corrected value
        all_groups = self._detections.groupby(self._JOIN).size().index
        
        # Create Series with False for all oids
        corrected_values = pd.Series(False, index=all_groups, name='corrected')
        
        # Return corrected values
        return pd.DataFrame({
            "corrected": corrected_values
        })
    
    # Since we haven't defined stellar for LSST detections, all objects are not stellar for now
    def calculate_stellar(self) -> pd.DataFrame:
        # Get all unique oids from detections so each oid in objstats will have a stellar value
        all_groups = self._detections.groupby(self._JOIN).size().index
        
        # Create Series with False for all oids
        stellar_values = pd.Series(False, index=all_groups, name='stellar')
        
        # Return stellar values
        return pd.DataFrame({
            "stellar": stellar_values
        })
    
    # Since we haven't defined dubious for LSST detections, all objects are not dubious for now
    def calculate_dubious(self) -> pd.DataFrame:
        # Get all unique oids from detections so each oid in objstats will have a dubious value
        all_groups = self._detections.groupby(self._JOIN).size().index
        
        # Create Series with False for all oids
        dubious_values = pd.Series(0, index=all_groups, name='stellar')
        
        # Return stellar values
        return pd.DataFrame({
            "ndubious": dubious_values
        })

register_survey_class_objstat("LSST", LSSTObjectStatistics)