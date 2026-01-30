from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class SurveyResults:
    """Container for all survey processing results"""
    magstats: List[pd.DataFrame]  # List of magstats DataFrames to concat
    objstats: Dict[str, Dict]     # Merged objstats by oid


class SurveyProcessor(ABC):
    def __init__(self, survey: str, excluded: set):
        self.survey = survey
        self.excluded = excluded
    
    @abstractmethod
    def extract_data(self, messages: List[Dict]) -> 'SurveyData':
        """Extract survey-specific data from messages"""
        pass
    
    @abstractmethod
    def process(self, survey_data: 'SurveyData') -> SurveyResults:
        """Process extracted data and return magstats + objstats"""
        pass
    
    def _calculate_magstats(self, detections, non_detections) -> pd.DataFrame:
        """Common magstats calculation logic"""
        from ..StatisticsSelector.statistics_selector import get_magnitude_statistics_class
        
        mag_stats_class = get_magnitude_statistics_class(self.survey)
        calculator = mag_stats_class(detections=detections, non_detections=non_detections)
        return calculator.generate_statistics(self.excluded).reset_index().set_index("oid").replace({np.nan: None})
    
    def _calculate_objstats(self, detections, non_detections, forced_detections) -> dict:
        """Common objstats calculation logic"""
        from ..StatisticsSelector.statistics_selector import get_object_statistics_class
        
        obj_stats_class = get_object_statistics_class(self.survey)
        calculator = obj_stats_class(detections, non_detections, forced_detections, self.survey)
        stats = calculator.generate_statistics(self.excluded).replace({np.nan: None})
        assert stats.index.name == "oid"
        return stats.to_dict("index")
    
    @staticmethod
    def _merge_objstats(target: dict, source: dict):
        """Merge objstats dictionaries - restructures to nested format by sid"""
        for oid, data in source.items():
            sid = data["sid"]
            target.setdefault(oid, {})
            target[oid][sid] = data