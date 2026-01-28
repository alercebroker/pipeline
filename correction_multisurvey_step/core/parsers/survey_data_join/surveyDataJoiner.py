import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any

class SurveyDataJoiner(ABC):
    """
    Abstract base class for survey-specific data joining strategies.
    
    It handles separation of historical data from detections to separate 
    into candidates/prv_candidates or detections into sources/prv_sources.
    Then it combines the message data with the processed historical data.
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
        Post-process combined data (sort, deduplicate, etc).
        
        Args:
            combined_data: Dictionary with combined dataframes
            
        Returns:
            Dictionary with post-processed dataframes
        """
        pass
    
    def process(self, parsed_data: Dict[str, Any], historical_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """
        Template method that orchestrates the complete joining process.
        
        This method calls the three abstract methods in the correct order,
        enforcing the pattern at the base class level.
        
        Args:
            parsed_data: Dict with 'data' key containing parsed message DataFrames
            historical_data: Dict with historical DataFrames from database
            
        Returns:
            Dict with combined and processed DataFrames ready for correction
        """
        # Extract message data
        msg_data = parsed_data['data']
        
        # Step 1: Process historical data according to survey rules
        processed_historical = self.process_historical_data(historical_data)
        
        # Step 2: Combine message data with processed historical data
        combined_data = self.combine_data(msg_data, processed_historical)
        
        # Step 3: Post-process (sort, deduplicate, etc.)
        result = self.post_process_data(combined_data)
        
        return result