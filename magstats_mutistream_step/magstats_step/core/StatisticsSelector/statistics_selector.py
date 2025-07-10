from typing import Dict, Type
from ..BaseStatistics.BaseMagstats import BaseMagnitudeStatistics
from ..BaseStatistics.BaseObjectstats import BaseObjectStatistics

class SurveySelector:    
    def __init__(self):
        self._object_stats: Dict[str, Type[BaseObjectStatistics]] = {}
        self._magnitude_stats: Dict[str, Type[BaseMagnitudeStatistics]] = {}

    def register_object_stats(self, survey: str, cls: Type[BaseObjectStatistics]):
        """Register an object statistics class for a survey"""
        self._object_stats[survey.upper()] = cls
    
    def register_magnitude_stats(self, survey: str, cls: Type[BaseMagnitudeStatistics]):
        """Register a magnitude statistics class for a survey"""
        self._magnitude_stats[survey.upper()] = cls
    
    def get_object_stats_class(self, survey: str) -> Type[BaseObjectStatistics]:
        """Get object statistics class for a survey"""
        cls = self._object_stats.get(survey.upper())
        if cls is None:
            raise ValueError(f"No object statistics class registered for survey: {survey}")
        return cls
    
    def get_magnitude_stats_class(self, survey: str) -> Type[BaseMagnitudeStatistics]:
        """Get magnitude statistics class for a survey"""
        cls = self._magnitude_stats.get(survey.upper())
        if cls is None:
            raise ValueError(f"No magnitude statistics class registered for survey: {survey}")
        return cls

# Global selector of statistic class
survey_registry = SurveySelector()

# Functions to access the statistic for current a survey, and to register magstats and objstats specific classes per survey

def get_object_statistics_class(survey: str) -> Type[BaseObjectStatistics]:
    return survey_registry.get_object_stats_class(survey)

def get_magnitude_statistics_class(survey: str) -> Type[BaseMagnitudeStatistics]:
    return survey_registry.get_magnitude_stats_class(survey)

def register_survey_class_objstat(survey: str, object_class: Type[BaseObjectStatistics]):
    survey_registry.register_object_stats(survey, object_class)

def register_survey_class_magstat(survey: str, magstat_class: Type[BaseMagnitudeStatistics]):
    survey_registry.register_magnitude_stats(survey, magstat_class)