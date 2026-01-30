from typing import List, Dict
from ....core.surveys.survey_processor import SurveyProcessor, SurveyResults
from ....core.parsers.survey_preparser import SurveyData, LSSTSurveyData  
from .pre_execute import extract_lsst_data


class LSSTSurveyProcessor(SurveyProcessor):
    """LSST-specific processing logic with SS detections support"""
    
    def extract_data(self, messages: List[Dict]) -> LSSTSurveyData:
        """Extract LSST data including SS detections - returns LSSTSurveyData"""
        detections, forced_photometries, non_detections, ss_detections = [], [], [], []
        
        for msg in messages:
            dets, f_phots, non_dets, ss_dets = extract_lsst_data(msg)
            detections.extend(dets)
            forced_photometries.extend(f_phots)
            non_detections.extend(non_dets)
            ss_detections.extend(ss_dets)
        
        # Returns extended LSSTSurveyData with SS detections
        return LSSTSurveyData(
            detections=detections,
            forced_photometries=forced_photometries,
            non_detections=non_detections,
            ss_detections=ss_detections
        )
    
    def process(self, survey_data: SurveyData) -> SurveyResults:
        """Process LSST data - handles both SS and regular detections"""
        magstats_list = []
        all_objstats = {}
        
        # Type check to ensure we have LSST-specific data
        if not isinstance(survey_data, LSSTSurveyData):
            raise TypeError(f"LSSHandler requires LSSTSurveyData, got {type(survey_data)}")
        
        # Process SS detections first
        if survey_data.ss_detections:
            ss_magstats = self._calculate_magstats(
                survey_data.ss_detections,
                survey_data.non_detections
            )
            ss_objstats = self._calculate_objstats(
                survey_data.ss_detections,
                survey_data.non_detections,
                survey_data.forced_photometries
            )
            magstats_list.append(ss_magstats)
            self._merge_objstats(all_objstats, ss_objstats)
        
        # Process regular detections
        if survey_data.detections:
            reg_magstats = self._calculate_magstats(
                survey_data.detections,
                survey_data.non_detections
            )
            reg_objstats = self._calculate_objstats(
                survey_data.detections,
                survey_data.non_detections,
                survey_data.forced_photometries
            )
            magstats_list.append(reg_magstats)
            self._merge_objstats(all_objstats, reg_objstats)
        
        return SurveyResults(magstats=magstats_list, objstats=all_objstats)
