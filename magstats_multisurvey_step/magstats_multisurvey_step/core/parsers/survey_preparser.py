from typing import List, Dict
from dataclasses import dataclass


@dataclass
class SurveyData:
    """Base survey data that all surveys must provide"""
    detections: List[Dict]
    forced_photometries: List[Dict]
    non_detections: List[Dict]


@dataclass
class LSSTSurveyData(SurveyData):
    """Extended LSST data with SS detections"""
    ss_detections: List[Dict] = None
    
    def __post_init__(self):
        if self.ss_detections is None:
            self.ss_detections = []
