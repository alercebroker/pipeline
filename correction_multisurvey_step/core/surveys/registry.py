"""
Centralized survey component registry.
This module provides a single point of configuration for all survey-specific components.
"""

from typing import Dict, Any
from core.DB.database_connection import PSQLConnection

from core.parsers.input_message_parsing.ztf_input_parser import ZTFInputMessageParser
from core.parsers.input_message_parsing.lsst_input_parser import LSSTInputMessageParser
from core.DB.ztf_database_strategy import ZTFDatabaseStrategy
from core.DB.lsst_database_strategy import LSSTDatabaseStrategy
from core.parsers.survey_data_join.ztfDataJoiner import ZTFDataJoiner
from core.parsers.survey_data_join.lsstDataJoiner import LSSTDataJoiner


# Registry mapping survey names to survey-specific component classes
SURVEY_COMPONENTS = {
    "ztf": {
        "input_parser_class": ZTFInputMessageParser,
        "db_strategy_class": ZTFDatabaseStrategy,
        "joiner_class": ZTFDataJoiner,
    },
    "lsst": {
        "input_parser_class": LSSTInputMessageParser,
        "db_strategy_class": LSSTDatabaseStrategy,
        "joiner_class": LSSTDataJoiner,
    },
}


def get_survey_components(survey_name: str, db_connection: PSQLConnection) -> Dict[str, Any]:
    """
    Get initialized survey-specific components.
    
    Args:
        survey_name: Survey identifier (case-insensitive)
        db_connection: Database connection instance
        
    Returns:
        Dictionary with initialized component instances:
            - input_parser: Parses survey-specific input messages
            - db_strategy: Queries survey-specific database tables
            - joiner: Joins message data with historical data using survey logic
            
    Raises:
        ValueError: If survey is not registered
    """
    survey_name = survey_name.lower()
    
    if survey_name not in SURVEY_COMPONENTS:
        supported = list(SURVEY_COMPONENTS.keys())
        raise ValueError(
            f"Unknown survey: '{survey_name}'. "
            f"Supported surveys: {supported}"
        )
    
    config = SURVEY_COMPONENTS[survey_name]
    
    components = {
        'input_parser': config['input_parser_class'](),
        'db_strategy': config['db_strategy_class'](db_connection),
        'joiner': config['joiner_class'](),
    }
    
    return components


def get_supported_surveys() -> list:
    """Return list of supported survey names."""
    return list(SURVEY_COMPONENTS.keys())