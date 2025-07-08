from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any
from .database_connection import PSQLConnection

class DatabaseStrategy(ABC):
    """
    Abstract strategy for database operations specific to each survey.
    
    Each survey has different database schemas, table structures, and parsing logic.
    This strategy encapsulates ALL database-related operations for a survey.
    """
    
    def __init__(self, db_connection: PSQLConnection):
        self.db_connection = db_connection
    
    @abstractmethod
    def get_detections(self, oids: List[str]) -> List[Dict[str, Any]]:
        """
        Query and parse detections from database for given OIDs.
        
        Args:
            oids: List of object IDs to query
            
        Returns:
            List of detection dictionaries with 'new': False flag added
        """
        pass
    
    @abstractmethod
    def get_forced_photometry(self, oids: List[str]) -> List[Dict[str, Any]]:
        """
        Query and parse forced photometry from database for given OIDs.
        
        Args:
            oids: List of object IDs to query
            
        Returns:
            List of forced photometry dictionaries with 'new': False flag added
        """
        pass
    
    @abstractmethod
    def get_non_detections(self, oids: List[str]) -> List[Dict[str, Any]]:
        """
        Query and parse non-detections from database for given OIDs.
        
        Args:
            oids: List of object IDs to query
            
        Returns:
            List of non-detection dictionaries
        """
        pass

    @abstractmethod
    def get_detection_schema(self) -> Dict[str, Any]:
        """Return the schema for detections specific to this survey."""
        pass
    
    @abstractmethod
    def get_forced_photometry_schema(self) -> Dict[str, Any]:
        """Return the schema for forced photometry specific to this survey."""
        pass
    
    @abstractmethod
    def get_non_detection_schema(self) -> Dict[str, Any]:
        """Return the schema for non-detections specific to this survey."""
        pass
    
    def get_all_historical_data(self, oids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all historical data types for given OIDs.
        
        This is the main method called by step.py. It returns a standardized
        structure that matches the survey's data organization.
        
        Args:
            oids: List of object IDs to query
            
        Returns:
            Dict with keys matching the survey's data structure
        """
        return {
            'detections': self.get_detections(oids),
            'forced_photometry': self.get_forced_photometry(oids),
            'non_detections': self.get_non_detections(oids)
        }

    def get_all_historical_data_as_dataframes(self, oids: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Get all historical data types for given OIDs as DataFrames with schemas applied.
        
        This is the main method that should be called by step.py for processed data.
        
        Args:
            oids: List of object IDs to query
            
        Returns:
            Dict with DataFrames for each data type, with proper schemas applied
        """
        from ..schemas.schema_applier import apply_schema
        
        # Get the data as dictionaries first
        raw_data = self.get_all_historical_data(oids)
        
        # Create a result dict to hold the processed data with proper schemas inside each key
        result = {}
        
        # Apply the schemas to detections
        result['detections'] = apply_schema(
            raw_data['detections'], 
            self.get_detection_schema()
        )
        
        # Apply the schemas to forced photometry
        result['forced_photometry'] = apply_schema(
            raw_data['forced_photometry'], 
            self.get_forced_photometry_schema()
        )
        
        # Apply the schemas to non-detections
        result['non_detections'] = apply_schema(
            raw_data['non_detections'], 
            self.get_non_detection_schema()
        )
        
        return result
    
    def _model_to_dict(self, model) -> Dict[str, Any]:
        """Convert SQLAlchemy model to dictionary, handling skipping the fields starting with _"""
        # This can be overridden by subclasses if needed
        result = {}
        for field, value in model.__dict__.items():
            if field.startswith("_"):
                continue
            result[field] =  value
        
        return result
    

def get_database_strategy(survey: str, db_connection: PSQLConnection) -> DatabaseStrategy:
    """
    Factory function to get the appropriate database strategy for a survey.
    
    Args:
        survey: Survey identifier ('lsst', 'ztf', etc.)
        db_connection: Database connection instance
        
    Returns:
        DatabaseStrategy: Concrete strategy for the survey
    """
    survey_lower = survey.lower()
    
    if survey_lower == 'lsst':
        from .lsst_database_strategy import LSSTDatabaseStrategy
        return LSSTDatabaseStrategy(db_connection)
    elif survey_lower == 'ztf':
        from .ztf_database_strategy import ZTFDatabaseStrategy
        return ZTFDatabaseStrategy(db_connection)
    else:
        supported_surveys = ['lsst', 'ztf']
        raise ValueError(f"Unknown survey: '{survey}'. Supported: {supported_surveys}")