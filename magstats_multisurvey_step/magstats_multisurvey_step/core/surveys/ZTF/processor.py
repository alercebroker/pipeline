from typing import List, Dict
from ....core.surveys.survey_processor import SurveyProcessor, SurveyResults
from ....core.parsers.survey_preparser import SurveyData 
from .pre_execute import extract_ztf_data


class ZTFSurveyProcessor(SurveyProcessor):
    """ZTF-specific processing logic"""
    
    def extract_data(self, messages: List[Dict]) -> SurveyData:
        """Extract ZTF data from messages - returns base SurveyData"""
        detections, forced_photometries, non_detections = [], [], []
        
        for msg in messages:
            dets, f_phots, non_dets = extract_ztf_data(msg)
            detections.extend(dets)
            forced_photometries.extend(f_phots)
            non_detections.extend(non_dets)
        
        return SurveyData(
            detections=detections,
            forced_photometries=forced_photometries,
            non_detections=non_detections
        )
    
    def process(self, survey_data: SurveyData) -> SurveyResults:
        """Process ZTF data - straightforward single detection type"""
        magstats_list = []
        all_objstats = {}
        
        if survey_data.detections:
            magstats = self._calculate_magstats(
                survey_data.detections, 
                survey_data.non_detections
            )
            objstats = self._calculate_objstats(
                survey_data.detections,
                survey_data.non_detections,
                survey_data.forced_photometries
            )
            magstats_list.append(magstats)
            self._merge_objstats(all_objstats, objstats)
        
        return SurveyResults(magstats=magstats_list, objstats=all_objstats)