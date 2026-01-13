from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any
from .database_connection import PSQLConnection
import logging

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
    
    def _model_to_dict(self, model) -> Dict[str, Any]:
        """Convert SQLAlchemy model to dictionary, handling skipping the fields starting with _"""
        # This can be overridden by subclasses if needed
        result = {}
        for field, value in model.__dict__.items():
            if field.startswith("_"):
                continue
            result[field] =  value
        
        return result
    

