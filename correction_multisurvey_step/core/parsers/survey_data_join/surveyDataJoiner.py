import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any

class SurveyDataJoiner(ABC):
    """Abstract base class for survey-specific data joining strategies.
    It handles separation of historical data from detections to separate 
    into candidates/prv_candidates or detections into sources/prv_sources.
    
    Then it combines the message data with the processed historical data
    """
    
    @abstractmethod
    def process_historical_data(self, historical_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Process historical data according to survey-specific rules.
        
        Args:
            historical_data: Dict with keys like 'detections', 'forced_photometry', 'non_detections'
            
        Returns:
            Dict with processed historical dataframes
        """
        pass
    
    @abstractmethod
    def combine_data(self, msg_data: Dict[str, pd.DataFrame], 
                    historical_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Combine message data with historical data.
        
        Args:
            msg_data: Dictionary containing message dataframes
            historical_data: Dictionary containing processed historical dataframes
            
        Returns:
            Dictionary with combined dataframes
        """
        pass

    @abstractmethod
    def post_process_data(self, combined_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Combine message data with historical data.
        
        Args:
            historical_data: Dictionary containing processed historical dataframes
            
        Returns:
            Dictionary with combined dataframes
        """
        pass
