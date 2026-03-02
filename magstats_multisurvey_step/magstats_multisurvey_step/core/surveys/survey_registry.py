from typing import Dict, Type
from .survey_processor import SurveyProcessor


class SurveyRegistry:
    """Registry for survey handlers"""
    
    _handlers: Dict[str, Type[SurveyProcessor]] = {}
    
    @classmethod
    def register(cls, survey: str, handler_class: Type[SurveyProcessor]):
        """Register a survey handler"""
        cls._handlers[survey.lower()] = handler_class
    
    @classmethod
    def get_handler(cls, survey: str, excluded: set) -> SurveyProcessor:
        """Get handler instance for a survey"""
        survey_lower = survey.lower()
        if survey_lower not in cls._handlers:
            raise ValueError(
                f"Unsupported survey: {survey}. "
                f"Supported surveys: {list(cls._handlers.keys())}"
            )
        return cls._handlers[survey_lower](survey_lower, excluded)


# Register handlers
from ..surveys.ZTF.processor import ZTFSurveyProcessor
from ..surveys.LSST.processor import LSSTSurveyProcessor

SurveyRegistry.register("ztf", ZTFSurveyProcessor)
SurveyRegistry.register("lsst", LSSTSurveyProcessor)