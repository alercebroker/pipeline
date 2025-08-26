import pandas as pd
from typing import Dict, Any
from .surveyDataJoiner import SurveyDataJoiner 
from .lsstDataJoiner import LSSTDataJoiner
from .ztfDataJoiner import ZTFDataJoiner

class SurveyDataProcessor:
    """Main processor that handles different survey data joining strategies"""
    
    def __init__(self, logger=None):
        self.joiners = {
            'lsst': LSSTDataJoiner(),
            'ztf': ZTFDataJoiner(),
            }
    
    def register_joiner(self, survey_name: str, joiner: SurveyDataJoiner):
        self.joiners[survey_name] = joiner
    
    def process_survey_data(self, survey_name: str, parsed_data: Dict[str, Any], 
                          historical_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Process survey data using the appropriate strategy.
        
        Args:
            survey_name: Name of the survey ('lsst', 'ztf', etc.)
            parsed_data: Dictionary containing parsed message data
            historical_data: Dictionary containing historical data
            
        Returns:
            Dictionary with combined dataframes according to survey strategy
        """
        if survey_name not in self.joiners:
            raise ValueError(f"No joiner registered for survey: {survey_name}")
        
        # Get the appropriate joiner for the current survey
        joiner = self.joiners[survey_name]
        
        # Extract the data from the messages
        msg_data = parsed_data['data']
        
        # Pre process the data from the historical data according to survey strategy
        processed_historical = joiner.process_historical_data(historical_data)
        
        # Combine data according to survey strategy
        combined_data = joiner.combine_data(msg_data, processed_historical)
        

        # Post-process data, by sorting and deduplicating to keep the newest data
        result = joiner.post_process_data(combined_data)
        
        return result