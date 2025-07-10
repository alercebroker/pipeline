from __future__ import annotations

import importlib
import logging
from typing import Any

import numpy as np
import pandas as pd
import copy

# Specific survey list of keys to be corrected
# Corrector receives a parsed data dict, which contains specific keys per surveys
# We use this dict to select only the corresponding keys that have detections to be corrected
SURVEY_DETECTION_KEYS = {
    "ztf": ["candidates", "previous_candidates", "forced_photometries"]
}

class StrategySelector:
    """Factory class for selecting correction strategies"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"alerce.{self.__class__.__name__}")
        self._strategies = {}
        self._no_correction_surveys = set()
    
    def get_strategy(self, survey: str) -> Any | None:
        """
        Get correction strategy for a survey, and returns the module if the survey has a strategy.
        None if no corrections are to be done for the survey, so the survey is marked as a do nothing by the corrector
        """
        
        # Check if this survey is marked as a survey with no corrections strategy
        if survey in self._no_correction_surveys:
            return None
        
        # Check if we already have this strategy cached
        if survey in self._strategies:
            return self._strategies[survey]
        
        # Try to import strategy module
        try:
            strategy_module = importlib.import_module(f"core.Corrector.strategy.{survey}")
            
            # Check if it's a no-corrections strategy or if it doesnt have any module for strategy, it considers it a no corrections survey
            if getattr(strategy_module, 'NO_CORRECTIONS', False):
                self.logger.info(f"Survey '{survey}' marked as a survey wit no corrections")
                self._no_correction_surveys.add(survey)
                return None
            
            # Validate that the strategy has the required functions for correction of the detections
            required_functions = ['is_corrected', 'is_dubious', 'is_stellar', 'correct']
            missing_functions = [func for func in required_functions if not hasattr(strategy_module, func)]
            
            # If there's missing functions, then the survey has to be considered a no corrections survey
            if missing_functions:
                self.logger.error(f"Strategy module for '{survey}' missing required functions: {missing_functions}")
                self._no_correction_surveys.add(survey)
                return None
            
            # Correctly loaded the module with all the required functions 
            self.logger.info(f"Loaded correction strategy for {survey}")
            self._strategies[survey] = strategy_module
            return strategy_module
                
        # Error in the imports mean that the survey cannot apply all the corrections from the corrector, so its going to pass as a no corrections survey
        except ImportError:
            self.logger.info(f"No strategy module found for survey '{survey}', treating as no-corrections survey")
            self._no_correction_surveys.add(survey)
            return None
    
    def has_corrections(self, survey: str) -> bool:
        """Check if a survey has corrections available"""
        return self.get_strategy(survey) is not None
    


class Corrector:
    """Class to apply corrections to detections in surveys"""

    _ZERO_MAG = 100.0

    def __init__(self, parsed_data: dict[str, pd.DataFrame]):
        """Init corrector.

        Args:
            parsed_data: Dict with historical data + message data. Each key is a dataframe)
            strategy_selector
        """
        self.logger = logging.getLogger(f"alerce.{self.__class__.__name__}")
        self.parsed_data = copy.deepcopy(parsed_data)
        self.strategy_selector = StrategySelector()
        self._strategy = None
        self._detection_keys = []
        self._survey = None
        self._has_corrections = False

    def set_survey(self, survey: str):
        """Set the survey to use and selects and imports its correction strategy.
        
        Args:
            survey: Survey name string used to identify strategy and survey
        
        Returns:
            self: Returns self for method chaining
        """
        self._survey = survey
        self._strategy = self.strategy_selector.get_strategy(survey)
        self._detection_keys = SURVEY_DETECTION_KEYS.get(survey.lower(), [])
        self._has_corrections = self._strategy is not None
        
        if self._has_corrections:
            self.logger.info(f"Set survey to '{survey}' with corrections for keys: {self._detection_keys}")
        else:
            self.logger.info(f"Set survey to '{survey}' with no corrections - will return original data")
        
        return self

    def _apply_strategy_function(self, df: pd.DataFrame, function_name: str, default=None, columns=None, dtype=object):
        """Applies given function from the strategy to a single DataFrame"""
        
        # Handle no detections dataframe. Separate because corrected returns a dataframe, while dubious, stellar and corrected return a series
        if df.empty:
            return pd.DataFrame(columns=columns, dtype=dtype) if columns else pd.Series(dtype=dtype)

        # Create a pre-filled dataframe to fill with the results' default value
        if columns:
            result = pd.DataFrame(default, index=df.index, columns=columns, dtype=dtype)
        else:
            result = pd.Series(default, index=df.index, dtype=dtype)
        
        # Apply the strategy function to the original dataframe
        function_output = getattr(self._strategy, function_name)(df)

        # Replace in the pre-filled dataframe the actual calculated values 
        result[:] = function_output.loc[df.index].astype(dtype).values
        
        return result.astype(dtype)

    def _correct_dataframe_inplace(self, df: pd.DataFrame):
        """Apply corrections to a single dataframe. Must be applied for each key in the survey detection keys"""
        # If there's no detection to correct, then there's no need to apply correction so we end the correction for this key
        if len(df) == 0:
            return
        
        # Calculate corrected, dubious and stellar for all detections in the current key
        corrected = self._apply_strategy_function(df, "is_corrected", default=False, dtype=bool)
        dubious = self._apply_strategy_function(df, "is_dubious", default=False, dtype=bool)
        stellar = self._apply_strategy_function(df, "is_stellar", default=False, dtype=bool)
        
        # Get corrected magnitudes
        cols = ["mag_corr", "e_mag_corr", "e_mag_corr_ext"]
        corrected_mags = self._apply_strategy_function(df, "correct", columns=cols, dtype=float)
        
        # Replace the magnitudes only for detections where corrected=True
        corrected_mags = corrected_mags.where(corrected)
        corrected_mags = corrected_mags.replace(np.inf, self._ZERO_MAG)
        
        # Modify columns directly to the original DataFrame of the detectionand replace nan so they won't cause errors 
        df["mag_corr"] = corrected_mags["mag_corr"].replace(np.nan, None)
        df["e_mag_corr"] = corrected_mags["e_mag_corr"].replace(np.nan, None)
        df["e_mag_corr_ext"] = corrected_mags["e_mag_corr_ext"].replace(np.nan, None)
        df["corrected"] = corrected
        df["dubious"] = dubious
        df["stellar"] = stellar
        
        # Replace infinity same as original Corrector
        # Theres no return because the modification is done in place so the Corrector's self.parsed data key must not be modified
        df.replace(-np.inf, None, inplace=True)

    def corrected_as_dataframes(self) -> dict[str, pd.DataFrame]:
        """Apply corrections to the original dict of parsed data for survey and return as dict of DataFrames
        with the detection keys corrected if the current survey has correction strategy, or unmodified if not.

        Returns:
            dict: Dictionary with same keys as input, but with corrected DataFrames
        """
        if self._survey is None:
            raise RuntimeError("Must call set_survey(survey) first to initialize strategy")
        
        # If no corrections are available for the survey, return original data without modifications
        if not self._has_corrections:
            return self.parsed_data
        
        # Apply corrections to detection DataFrames for each of the survey detection keys
        for key in self.parsed_data.keys():
            if key in self._detection_keys:
                df = self.parsed_data[key]
                self.logger.info(f"Processing {len(df)} detections from {key}")
                
                # Modify the detections in place of the parsed data key
                self._correct_dataframe_inplace(df)
                
                num_corrected = df["corrected"].sum() if len(df) > 0 else 0
                if num_corrected > 0:
                    self.logger.info(f"Corrected {num_corrected} detections in {key}")
                else:
                    self.logger.info(f"No corrections applied to {key}")


        
        # Return the parsed data with the corrections applied
        return self.parsed_data

    def has_corrections_for_survey(self) -> bool:
        """Check if the current survey has corrections available"""
        return self._has_corrections
    

