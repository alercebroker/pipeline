from typing import List, Dict, Tuple, NamedTuple
from ..surveys.ZTF.pre_execute import extract_ztf_data
from ..surveys.LSST.pre_execute import extract_lsst_data


# Container for survey data extractions
class SurveyData(NamedTuple):
    detections: List[Dict]
    forced_photometries: List[Dict]
    non_detections: List[Dict]
    ss_detections: List[Dict] = []  # SS detections (LSST only!)

# Registry of all available extractors
SURVEY_EXTRACTORS = {
    "ztf": extract_ztf_data,
    "lsst": extract_lsst_data,
}

class SurveyDataSelector:
    """Simple survey data selector that uses separate extractor functions to fuse together detections/sources to execute magstats to all data"""
    
    def __init__(self, survey: str, config: Dict = None):
        self.survey = survey.lower()
    
    def extract_data(self, messages: List[Dict]) -> SurveyData:
        """Extract all detection types based on survey."""
        detections, forced_photometries, non_detections, ss_detections = [], [], [], []
        
        for msg in messages:
            if self.survey in SURVEY_EXTRACTORS:
                extracted = SURVEY_EXTRACTORS[self.survey](msg)
                
                # Handle surveys with and without ss_detections
                if len(extracted) == 4:
                    dets, f_phots, non_dets, ss_dets = extracted
                    ss_detections.extend(ss_dets)
                else:
                    dets, f_phots, non_dets = extracted
                    
                detections.extend(dets)
                forced_photometries.extend(f_phots)
                non_detections.extend(non_dets)
            else:
                raise ValueError(
                    f"Unsupported survey: {self.survey}. "
                    f"Supported surveys: {list(SURVEY_EXTRACTORS.keys())}"
                )
        
        return SurveyData(
            detections=detections,
            forced_photometries=forced_photometries,
            non_detections=non_detections,
            ss_detections=ss_detections
        )