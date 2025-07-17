from typing import List, Dict, Tuple
from ..surveys.ZTF.pre_execute import extract_ztf_data
from ..surveys.LSST.pre_execute import extract_lsst_data

# Registry of all available extractors
SURVEY_EXTRACTORS = {
    "ztf": extract_ztf_data,
    "lsst": extract_lsst_data,
}

class SurveyDataSelector:
    """Simple survey data selector that uses separate extractor functions to fuse together detections/sources to execute magstats to all data"""
    
    def __init__(self, survey: str, config: Dict = None):
        self.survey = survey
    
    def extract_data(self, messages: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Extract detections and non_detections based on survey strategy and data structure of the messages."""
        detections, forced_photometries, non_detections = [], [], []
        
        for msg in messages:
            if self.survey in SURVEY_EXTRACTORS:
                # Use specific extractor function that corresponds to current survey
                dets, f_phots, non_dets = SURVEY_EXTRACTORS[self.survey](msg)
            else:
                raise ValueError(f"Unsupported survey: {self.survey}. Supported surveys: {list(SURVEY_EXTRACTORS.keys())}")
            
            detections.extend(dets)
            forced_photometries.extend(f_phots)
            non_detections.extend(non_dets)
        
        return detections, forced_photometries, non_detections